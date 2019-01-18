import argparse, os, sys, logging, random, time, queue, signal, copy
import numpy as np
import torch
import torch.multiprocessing as mp

from option import opt
import process as proc
import utility as util

NUM_ITER = 1
MAX_FPS =  30
MAX_SEGMENT_LENGTH = 4
SHARED_QUEUE_LEN = MAX_FPS * MAX_SEGMENT_LENGTH #Regulate GPU memory usage (> 3 would be fine)

"""code for Figure 16
CLOCK_INFO = {}
#TITAN_XP_INFO = [1404, 1303, 1202, 1101, 999, 898, 797, 696, 506]
#TITAN_XP_INFO = [1404, 1303, 1202, 1101, 999, 898, 797, 696]
TITAN_XP_INFO = [949]
TITAN_XP_INFO.reverse()
CLOCK_INFO['titanxp'] = TITAN_XP_INFO

for clock in CLOCK_INFO['titanxp']:
    os.system('echo ina8024 | sudo -S nvidia-smi -i 0 --applications-clocks=5705,{}'.format(clock))
    test_figure16(clock)
os.system('echo ina8024 nvidia-smi -i 0 --reset-applications-clocks')
"""

if __name__ == "__main__":
    mp.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_descriptor')

    #create Queue, Pipe
    decode_queue = mp.Queue()
    dnn_queue = mp.JoinableQueue()
    data_queue = mp.JoinableQueue()
    encode_queue = mp.JoinableQueue()
    output_output, output_input = mp.Pipe(duplex=False)

    #create shared tensor
    shared_tensor_list = {}
    res_list = [(270, 480), (360, 640), (540, 960), (1080, 1920)]
    for res in res_list:
        shared_tensor_list[res[0]] = []
        for _ in range(SHARED_QUEUE_LEN):
            shared_tensor_list[res[0]].append(torch.ByteTensor(res[0], res[1], 3).cuda().share_memory_())

    #create processes
    decode_process = mp.Process(target=proc.decode, args=(decode_queue, encode_queue, data_queue, shared_tensor_list))
    sr_process = mp.Process(target=proc.super_resolution, args=(encode_queue, dnn_queue, data_queue, shared_tensor_list))
    encode_process = mp.Process(target=proc.encode, args=(encode_queue, shared_tensor_list))

    #start processes
    sr_process.start()
    decode_process.start()
    encode_process.start()

    pretrained_path = os.path.join(opt.checkpoint_dir, 'epoch_{}.pth'.format(opt.test_num_epoch))
    dnn_queue.put(('load_model', pretrained_path))
    dnn_queue.join()

    #caution: fps and (segment) duration should be given correctly
    segment_fps = 24
    segment_size = 4
    resolution_list = [240, 360, 480, 720]
    res2quality = {240: 0, 360: 1, 480: 2, 720: 3, 1080: 4}
    index = 1

    #execute dummy jobs
    for resolution in resolution_list:
        video_dir = os.path.join(opt.data_dir, '{}p'.format(resolution))
        video_info = util.videoInfo(segment_fps, segment_size, res2quality[resolution])
        output_output, output_input = mp.Pipe(duplex=False)
        decode_queue.put((os.path.join(video_dir, 'segment_init.mp4'), os.path.join(video_dir, 'segment_{}.m4s'.format(index)), output_input, video_info))

        while(1):
            input = output_output.recv()
            if input[0] == 'output':
                break
            else:
                print('request: Invalid input')
                break

    #iterate multiple time and get the average latency
    elapsed_time_list = {}
    fps_list = {}

    output_list = [0,1,2,3,4]
    for output in output_list:
        elapsed_time_list[output] = {}
        fps_list[output] = {}
        for resolution in resolution_list:
            elapsed_time_list[output][resolution] = []
            fps_list[output][resolution] = []

    #set inference index
    for _ in range(NUM_ITER):
        for output in output_list:
            dnn_queue.put(('set_inference_idx',output,))
            dnn_queue.join()
            for resolution in resolution_list:
                video_dir = os.path.join(opt.data_dir, '{}p'.format(resolution))
                video_info = util.videoInfo(segment_fps, segment_size, res2quality[resolution])
                output_output, output_input = mp.Pipe(duplex=False)
                start_time = time.time()
                decode_queue.put((os.path.join(video_dir, 'segment_init.mp4'), os.path.join(video_dir, 'segment_{}.m4s'.format(index)), output_input, video_info))
                while(1):
                    input = output_output.recv()
                    if input[0] == 'output':
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        fps = segment_fps * segment_size / (end_time - start_time)
                        print('overall [elapsed], resolution [{}p] : {} second, {} fps'.format(resolution, elapsed_time, fps))
                        elapsed_time_list[output][resolution].append(elapsed_time)
                        fps_list[output][resolution].append(fps)
                        break
                    else:
                        print('request: Invalid input')
                        break

    #print statistics
    runtimeLogger = util.get_logger(opt.result_dir, 'result_video_scalable_runtime.log')
    for output in output_list:
        for resolution in resolution_list:
            print('[output: {}][{}p]: minmum {} fps, average {} fps, maximum {} fps'.format(output, resolution, np.min(fps_list[output][resolution]), np.average(fps_list[output][resolution]), np.max(fps_list[output][resolution])))
            log_str = "\t".join(map(str, fps_list[output][resolution]))
            runtimeLogger.info(log_str)

    #terminate processes
    sr_process.terminate()
    decode_process.terminate()
    encode_process.terminate()
