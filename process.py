import argparse, os, sys, logging, random, time, queue, signal, copy, threading
from subprocess import Popen
import subprocess as sub
from shutil import copyfile
import numpy as np
from skimage.io import imsave
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
import torch.multiprocessing as mp
import cv2 #Import Error when import cv2 before torch

import utility as util
from option import opt
from model import MultiNetwork
import template

INPUT_VIDEO_NAME = 'input.mp4'
MAX_FPS =  30
MAX_SEGMENT_LENGTH = 4
SHARED_QUEUE_LEN = MAX_FPS * MAX_SEGMENT_LENGTH #Regulate GPU memory usage (> 3 would be fine)

"""Summary
Process
1. decode (1 queue: decode_queue)
2. super_resolution (2 threads: load_dnn_chunks, process_video_chunk, 2 queue: dnn_queue, data_queue )
3. encode (1 queue: encode_queue)
"""

def decode(decode_queue, encode_queue, data_queue, shared_tensor_list):
    while True:
        try:
            #0. read queue & setup
            input = decode_queue.get()
            start_time = time.time()

            header_file = input[0]
            video_file = input[1]
            output_input = input[2]
            video_info = input[3]

            if not os.path.exists(header_file):
                print('decode: header does not exist')
                continue
            if not os.path.exists(video_file):
                print('decode: video does not exist')
                continue

            video_file_name, _  = os.path.splitext(os.path.basename(video_file))
            process_dir = os.path.join(opt.result_dir, '{}_{}'.format(video_file_name, video_info.quality))
            os.makedirs(process_dir, exist_ok=True)

            #1. merge 'Header.m4s' and 'X.m4s' for decoding
            input_video = os.path.join(process_dir, INPUT_VIDEO_NAME)
            with open(input_video, 'wb')as outfile:
                with open(header_file, 'rb') as infile:
                    outfile.write(infile.read())

                with open(video_file, 'rb') as infile:
                    outfile.write(infile.read())

            #2. setup super-resolution, encode processes
            t_h, t_w = get_resolution(video_info.quality)
            target_scale = int(1080/t_h)
            target_height = t_h
            encode_queue.put(('start', process_dir, video_info))
            encode_queue.join()
            data_queue.put(('configure', target_scale, target_height))
            data_queue.join()
            print('decode [configuration]: {}sec'.format(time.time() - start_time))

            #3. read frames from a video and prepare (shared) PyTorch CUDA tensors
            vc = cv2.VideoCapture(input_video)
            frame_count = 0
            print('decode [video read prepare]: {}sec'.format(time.time() - start_time))
            while True:
                rval, frame = vc.read()
                if rval == False:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (t_w, t_h), interpolation=cv2.INTER_CUBIC) #add bicubic resize
                input_t_ = torch.from_numpy(frame).byte().cuda()

                #4. start super-resolution
                shared_tensor_list[t_h][frame_count % SHARED_QUEUE_LEN].copy_(input_t_)
                data_queue.put(('apply_sr', frame_count))
                frame_count += 1
            vc.release()

            print('decode [prepare_frames-{}]: {}sec'.format(frame_count, time.time() - start_time))
            #5. wait super-resolution, encode processes to be end
            data_queue.join()
            print('decode [super-resolution]: {}sec'.format(time.time() - start_time))
            encode_queue.join()
            encode_queue.put(('end', output_input))
            encode_queue.join()
            print('decode [encode] : {}sec'.format(time.time() - start_time))

        except (KeyboardInterrupt, SystemExit):
            print('exiting...')
            break

#Share following variables between multi-threads
inference_idx = 0
model = MultiNetwork(template.get_nas_config(opt.quality))
model = model.to(torch.device('cuda'))
model = model.half().to('cuda') #TODO: save the final model as half-precision

def load_dnn_chunk(dnn_queue):
    global inference_idx
    global model
    while True:
        try:
            input = dnn_queue.get()

            #load a pretrained model of which path is given
            if input[0] == 'load_model':
                start_time = time.time()

                pretrained_path = input[1]
                if not os.path.exists(pretrained_path):
                    print('sr: Model does not exist')
                    continue

                model.load_state_dict(torch.load(pretrained_path))
                scale_list = [1,2,3,4]
                for scale in scale_list:
                    model.setTargetScale(scale)
                    inference_idx = max(inference_idx, len(model.getOutputNodes())-1)
                elapsed_time = time.time()
                print('model_loading [elapsed] : {}sec'.format(elapsed_time - start_time))

            #set an inference index of a model
            elif input[0] == 'set_inference_idx':
                inference_idx = input[1]

            #set an inference index to use a full model
            elif input[0] == 'set_inference_idx_max':
                scale_list = [1,2,3,4]
                for scale in scale_list:
                    model.setTargetScale(scale)
                    inference_idx = max(inference_idx, len(model.getOutputNodes())-1)

            #load a DNN chunk
            elif input[0] == 'load_dnn_chunk':
                start_time = time.time()
                dnn_chunk_path = input[1]
                dnn_chunk_idx = input[2] #idx : 0,1,2,3,4

                weights = torch.load(dnn_chunk_path)
                model.load_state_dict(weights, strict=False)
                inference_idx = dnn_chunk_idx
                print('inference_idx: {}'.format(inference_idx))
                end_time = time.time()
                print('dnn_chunk_loading [elapsed] : {}sec'.format(end_time - start_time))

            #input: DNN_list, fps, duration
            elif input[0] == 'test_dnn':
                DNN_list = input[1]
                fps = input[2]
                duration = input[3]
                test_input = input[4]
                is_break = False

                #Prepare a random input
                start_time = time.time()
                random_tensor = torch.HalfTensor(1, 3, 480, 270) #TODO: test with resolution which takes the longest time
                input = Variable(random_tensor, volatile=True)
                input = input.cuda()

                #Prepare and test a mock DNN
                for DNN in DNN_list:
                    inference_time_list = []
                    layer = DNN[0]
                    feature = DNN[1]
                    #mock_DNN = NAS.Single_Network(nLayer=layer, nFeatBase=feature // 2, nFeatBody=feature, nChannel=3, scale=3, outputFilter=1, bias=True, act=nn.ReLU(True))
                    mock_DNN = NAS.Single_Network(nLayer=layer, nFeat=feature, nChannel=3, scale=4, outputFilter=2, bias=True, act=nn.ReLU(True))
                    mock_DNN = mock_DNN.cuda()
                    mock_DNN = mock_DNN.half()
                    output_node = mock_DNN.getOutputNodes()[-1]

                    #Dummy-inference: initial CUDA run has overhead
                    for _ in range(10):
                        mock_DNN(input, output_node)
                    torch.cuda.synchronize()

                    #Real-inference
                    for _ in range(10):
                        start_inference = time.time()
                        mock_DNN(input, output_node)
                        torch.cuda.synchronize()
                        end_inference = time.time()
                        inference_time_list.append(end_inference - start_inference)

                        if np.mean(inference_time_list) < (duration) / (fps * duration): #TODO: replace 0.1 with reasonable value
                            break


                    print('DNN inference [{}]: {}sec'.format(DNN_list.index(DNN),np.mean(inference_time_list)))
                    #if end_inference - start_inference > (duration - 0.1) / fps: #TODO: replace 0.1 with reasonable value

                    if np.mean(inference_time_list) > (duration) / (fps * duration): #TODO: replace 0.1 with reasonable value
                        is_break = True
                        break

                end_time = time.time()
                print('test mock DNNs [elapsed] : {}sec'.format(end_time - start_time))
                if is_break:
                    test_input.send((DNN_list.index(DNN) - 1,))
                else:
                    test_input.send((DNN_list.index(DNN),))

            else:
                print('sr: Invalid input')

            dnn_queue.task_done()

        except (KeyboardInterrupt, SystemExit):
            print('exiting...')
            break

def get_resolution(quality):
    assert quality in [0, 1, 2, 3]

    if quality == 3:
        t_w = 1920
        t_h = 1080
    elif quality == 2:
        t_w = 960
        t_h = 540
    elif quality == 1:
        t_w = 640
        t_h = 360
    elif quality == 0:
        t_w = 480
        t_h = 270

    return (t_h, t_w)

def process_video_chunk(encode_queue, shared_tensor_list, data_queue):
    global inference_idx
    global model
    target_height = None

    while True:
        try:
            input = data_queue.get()

            #setup
            if input[0] == 'configure':
                target_scale = input[1]
                target_height = input[2]
                model.setTargetScale(target_scale)
                if target_scale != 1:
                    inference_idx_ = inference_idx * 2
                else:
                    inference_idx_ = min(inference_idx, len(model.getOutputNodes())-1)
                inference_time_list = []
                encode_queue.put(('index', inference_idx_))

            #apply super-resolution
            elif input[0] == 'apply_sr':
                with torch.no_grad():
                    frame_count = input[1]

                    input_tensor_ = shared_tensor_list[target_height][frame_count % SHARED_QUEUE_LEN]
                    input_tensor_ = input_tensor_.permute(2,0,1).half()
                    input_tensor_.div_(255) #byte tensor/255
                    input_tensor_.unsqueeze_(0)
                    input_ = Variable(input_tensor_)

                    output_ = model(input_, inference_idx_)
                    output_ = output_.data[0].permute(1,2,0)
                    output_ = output_ * 255
                    output_ = torch.clamp(output_, 0, 255)
                    output_ = output_.byte()
                    shared_tensor_list[1080][frame_count % SHARED_QUEUE_LEN].copy_(output_)
                    torch.cuda.synchronize()

                    encode_queue.put(('encode', frame_count % SHARED_QUEUE_LEN))

                    """
                    inference_time_list.append(end_time - start_time)

                    #For measuring a DNN run-time
                    if frame_count == 119:
                        print('process [index: {}, total-{}frames]: {}sec'.format(inference_idx_, len(inference_time_list), np.sum(inference_time_list)))
                    """
            else:
                print('sr: Invalid input')

            data_queue.task_done()

        except (KeyboardInterrupt, SystemExit):
            print('exiting...')
            break

def super_resolution(encode_queue, dnn_queue, data_queue, shared_tensor_list):
    dnn_load_thread = threading.Thread(target=load_dnn_chunk, args=(dnn_queue,))
    video_process_thread = threading.Thread(target=process_video_chunk, args=(encode_queue, shared_tensor_list, data_queue))
    dnn_load_thread.start()
    video_process_thread.start()
    dnn_load_thread.join()
    video_process_thread.join()

def encode(encode_queue, shared_tensor_list):
    pipe = None
    process_dir = None
    infer_idx = None

    while(1):
        try:
            input = encode_queue.get()

            #setup
            if input[0] == 'start':
                encode_start_time = time.time()
                #print('encode [start]: {}sec'.format(encode_start_time))

                process_dir = input[1]
                video_info = input[2]

                fps = video_info.fps
                duration = video_info.duration
                total_frames = duration * fps

                print('encode [after video info]: {}sec'.format(time.time() - encode_start_time))

                command = [ 'ffmpeg',
                            '-r', str(fps), # frames per second
                            '-y',
                            '-loglevel', 'error',
                            '-f', 'rawvideo',
                            '-vcodec','rawvideo',
                            '-s', '1920x1080', # size of one frame
                            #'-s', '1280x720', # size of one frame
                            '-pix_fmt', 'rgb24',
                            '-i', '-', # The imput comes from a pipe
                            #'-s', '1920x1080', # size of one frame
                            '-vcodec', 'libx264',
                            #'crf', '0',
                            '-preset', 'ultrafast',
                            '-movflags', 'empty_moov+omit_tfhd_offset+frag_keyframe+default_base_moof',
                            '-pix_fmt', 'yuv420p',
                            #'-an', # Tells FFMPEG not to expect any audio
                            '{}'.format(os.path.join(process_dir, 'output.mp4'))]

                pipe = sub.Popen(command, stdin=sub.PIPE, stderr=sub.PIPE, stdout=sub.PIPE, bufsize=10**9)
                end_time_ = time.time()
                print('encode [start]: {}sec'.format(end_time_ - encode_start_time))

            #encode
            elif input[0] == 'encode':
                #start_time_ = time.time()
                idx = input[1]
                img = shared_tensor_list[1080][idx].cpu().numpy()

                if img is None:
                    print(idx)

                pipe.stdin.write(img.tobytes())
                pipe.stdin.flush()
                #end_time_ = time.time()
                #print('encode [frame]: {}sec'.format(end_time_ - start_time_))

            #save as a video
            elif input[0] == 'end':
                start_time_ = time.time()
                pipe.stdin.flush()
                pipe.stdin.close()
                pipe = None
                output_input = input[1]
                #infer_idx = input[2]
                #infer_idx = -1 #TODO

                #print('encode [end] : {}sec'.format(end_time))
                encode_end_time = time.time()
                print('encode [end]: {}sec'.format(encode_end_time - start_time_))
                print('encode [elapsed] / index [{}]: {}sec'.format(infer_idx, encode_end_time - encode_start_time))

                output_input.send(('output', os.path.join(process_dir, 'output.mp4'), infer_idx))
                process_dir = None


            elif input[0] == 'index':
                infer_idx = input[1]

            else:
                print('encode: Invalid input')
                continue

            encode_queue.task_done()

        except (KeyboardInterrupt, SystemExit):
            print('exiting...')
            break

#building a request
def request(decode_queue, resolution, index, fps=24.0, duration=4.0):
    res2quality = {240: 0, 360: 1, 480: 2, 720: 3, 1080: 4}
    video_dir = os.path.join(opt.data_dir, '{}p'.format(resolution))

    start_time = time.time()
    video_info = util.videoInfo(fps, duration, res2quality[resolution])
    output_output, output_input = mp.Pipe(duplex=False)
    decode_queue.put((os.path.join(video_dir, 'segment_init.mp4'), os.path.join(video_dir, 'segment_{}.m4s'.format(index)), output_input, video_info))

    while(1):
        input = output_output.recv()
        if input[0] == 'output':
            end_time = time.time()
            print('overall [elapsed], resolution [{}p] : {}sec'.format(resolution, end_time - start_time))
            break
        else:
            print('request: Invalid input')
            break

def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)
