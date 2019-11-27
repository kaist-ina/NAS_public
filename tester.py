import argparse, os, sys, logging, random, bisect, threading, queue, time
from datetime import datetime
import numpy as np
import multiprocessing as mp
from skimage.measure import compare_ssim
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

import utility as util
from option import opt
from model import MultiNetwork
from dataset import DatasetForDASH
import template
from scipy import misc

RESOLUTION = {240: (480, 270), 360: (640, 360), 480: (960, 540), 720: (1920, 1080), 1080: (1920, 1080)}
DUMMY_TRIAL = 3
TEST_TRIAL = 10
PROCESS_NUM = 4

"""Note
Measuring SSIM using sckit.image is quite slow.
Therefore, I implmented tester using multi-processes.
"""

class Quality: pass #class for recording result of analysis
class Result: pass #class for recording result of analysis

#measure psnr, ssim
def measure_quality(input_queue, output_queue):
    print(mp.current_process().name)
    quality_list = []
    print('process_start')
    while True:
        try:
            input = input_queue.get()
            if str(input[0]) == 'end':
                print('process end')
                break
            else:
                input_np = input[1]
                target_np = input[2]

                ssim = compare_ssim(input_np, target_np, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0, multichannel=True)
                psnr = util.get_psnr(input_np, target_np, max_value=1.0)

                quality = Quality()
                quality.idx = input[0]
                quality.ssim = ssim
                quality.psnr = psnr

                quality_list.append(quality)

        except (KeyboardInterrupt, SystemExit):
            print('exiting...')
            break

    output_queue.put(quality_list)

def save_img(input_queue):
    print(mp.current_process().name)
    print('process_start')
    while True:
        try:
            input = input_queue.get()
            if str(input[0]) == 'end':
                print('process end')
                break
            else:
                lr = input[0]
                frame_idx = input[1]
                input_np = input[2]
                output_np = input[3]
                target_np = input[4]

                misc.imsave('{}/{}_{}_input.png'.format(opt.result_dir, lr, frame_idx), input_np)
                misc.imsave('{}/{}_{}_output.png'.format(opt.result_dir, lr, frame_idx), output_np)
                misc.imsave('{}/{}_{}_target.png'.format(opt.result_dir, lr, frame_idx), target_np)

        except (KeyboardInterrupt, SystemExit):
            print('exiting...')
            break

class Tester:
    def __init__(self, opt, model, dataset):
        self.opt = opt
        self.model = model
        self.dataset = dataset
        self.dataset.setDatasetType('test')
        self.device = torch.device("cuda" if opt.use_cuda else "cpu")

        #load a model on a target device & Load weights
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load("{}/epoch_{}.pth".format(self.opt.checkpoint_dir, self.opt.test_num_epoch)))
        self.model.eval()

        #map resolution to outputNodes
        self.node2res = {}

        max_node = 0
        for res in opt.dash_lr:
            self.dataset.setTargetLR(res)
            self.model.setTargetScale(self.dataset.getTargetScale())
            nodes = self.model.getOutputNodes()

            for node in nodes:
                if node in self.node2res.keys():
                    self.node2res[node].append(res)
                else:
                    self.node2res[node] = []
                    self.node2res[node].append(res)

        self.output_nodes = list(self.node2res.keys())

    #get psnr, ssim of baseline methods (e.g., bicubic)
    def _analyze_baseline(self):
        timer = util.timer()
        dataloader = DataLoader(dataset=self.dataset, num_workers=self.opt.num_thread, batch_size=1, shuffle=False, pin_memory=True)
        result = {}

        #iterate over low resolutions
        for lr in opt.dash_lr:
            #setup (process & thread)
            process_list = []
            output_queue = mp.Queue()
            input_queue = mp.Queue(1)
            for _ in range(PROCESS_NUM):
                process = mp.Process(target=measure_quality, args=(input_queue, output_queue))
                process.start()
                process_list.append(process)

            #setup (variable)
            result[lr] = Result()
            result[lr].frameidx = []
            result[lr].ssim = []
            result[lr].psnr = []

            print('start anaylze {}p/baseline quality'.format(lr))
            self.dataset.setTargetLR(lr)
            self.model.setTargetScale(self.dataset.getTargetScale())

            #iterate over test dataset
            for iteration, batch in enumerate(dataloader, 1):
                assert len(batch[0]) == 1
                _, upscaled, target = batch[0], batch[1], batch[2]
                upscaled_np, target_np = torch.squeeze(upscaled, 0).permute(1, 2, 0).numpy(), torch.squeeze(target, 0).permute(1, 2, 0).numpy()
                upscaled_np= upscaled.data[0].permute(1,2,0).numpy()
                input_queue.put((iteration, upscaled_np, target_np))

                elapsed_time = timer.toc()
                util.print_progress(iteration, len(self.dataset), 'Test Progress ({}p - {}sec):'.format(lr, round(elapsed_time, 2)), 'Complete', 1, 50)


            #terminate
            for _ in range(len(process_list)):
                input_queue.put(('end', ))

            #merge results
            quality_list = []
            for process in process_list:
                quality_list.extend(output_queue.get())

            for quality in quality_list:
                result[lr].frameidx.append(quality.idx)
                result[lr].ssim.append(quality.ssim)
                result[lr].psnr.append(quality.psnr)

            result[lr].frameidx, result[lr].ssim, result[lr].psnr = \
                [list(x) for x in zip(*sorted(zip(result[lr].frameidx, result[lr].ssim, result[lr].psnr), key=lambda pair: pair[0]))]

        #PSNR, SSIM for original images
        result[self.opt.dash_hr] = Result()
        result[self.opt.dash_hr].psnr = 100
        result[self.opt.dash_hr].ssim = 1

        return result

    def _generate_sr(self, output_node=None):
        with torch.no_grad():
            timer = util.timer()
            if output_node == None :
                output_node = self.output_nodes[-1]

            dataloader = DataLoader(dataset=self.dataset, num_workers=6, batch_size=1, shuffle=False, pin_memory=True)
            target_res = self.node2res[output_node]
            
            process_list = []
            input_queue = mp.Queue()
            for _ in range(PROCESS_NUM):
                process = mp.Process(target=save_img, args=(input_queue, ))
                process.start()
                process_list.append(process)

            #iterate over target resolutions
            for lr in target_res:

                self.dataset.setTargetLR(lr)
                self.model.setTargetScale(self.dataset.getTargetScale())
                for iteration, batch in enumerate(dataloader, 1):
                    assert len(batch[0]) == 1

                    #prepare
                    input, upscaled, target = batch[0], batch[1], batch[2]
                    input_np, pscaled_np, target_np = torch.squeeze(input, 0).permute(1, 2, 0).numpy(), torch.squeeze(upscaled, 0).permute(1, 2, 0).numpy(), torch.squeeze(target, 0).permute(1, 2, 0).numpy()
                    input, upscaled, target =  input.to(self.device), upscaled.to(self.device), target.to(self.device)
                    output = self.model(input, output_node)
                    torch.cuda.synchronize()

                    output = torch.squeeze(torch.clamp(output, min=0, max=1.), 0).permute(1, 2, 0)
                    output_np = output.to('cpu').numpy()
                    '''
                    misc.imsave('{}/{}_{}_output.png'.format(self.opt.result_dir, lr, iteration), output_np)
                    misc.imsave('{}/{}_{}_baseline.png'.format(self.opt.result_dir, lr, iteration), upscaled_np)
                    misc.imsave('{}/{}_{}_target.png'.format(self.opt.result_dir, lr, iteration), target_np)
                    '''
                    input_queue.put((lr, iteration, input_np, output_np, target_np))
                    elapsed_time = timer.toc()
                    util.print_progress(iteration, len(self.dataset), 'Test Progress ({}p - {}sec):'.format(lr, round(elapsed_time, 2)), 'Complete', 1, 50)

            #terminate
            for _ in range(len(process_list)):
                input_queue.put(('end', ))

    #get psnr, ssim of super-resolution
    def _analyze_sr(self, output_node):
        with torch.no_grad():
            timer = util.timer()
            dataloader = DataLoader(dataset=self.dataset, num_workers=1, batch_size=1, shuffle=False, pin_memory=True)
            result = {}
            target_res = self.node2res[output_node]

            #iterate over target resolutions
            for lr in target_res:
                #setup (process & thread)
                process_list = []
                output_queue = mp.Queue()
                input_queue = mp.Queue(1)
                for _ in range(PROCESS_NUM):
                    process = mp.Process(target=measure_quality, args=(input_queue, output_queue))
                    process.start()
                    process_list.append(process)

                #setup (variable)
                result[lr] = Result()
                result[lr].frameidx = []
                result[lr].ssim = []
                result[lr].psnr = []

                print('start anaylze {}p/output node:{} quality'.format(lr, output_node))
                self.dataset.setTargetLR(lr)
                self.model.setTargetScale(self.dataset.getTargetScale())
                for iteration, batch in enumerate(dataloader, 1):
                    assert len(batch[0]) == 1

                    #prepare
                    input, upscaled, target = batch[0], batch[1], batch[2]
                    upscaled_np, target_np = torch.squeeze(upscaled, 0).permute(1, 2, 0).numpy(), torch.squeeze(target, 0).permute(1, 2, 0).numpy()
                    input, upscaled, target =  input.to(self.device), upscaled.to(self.device), target.to(self.device)
                    output = self.model(input, output_node)
                    torch.cuda.synchronize()

                    output = torch.squeeze(torch.clamp(output, min=0, max=1.), 0).permute(1, 2, 0)
                    output_np = output.to('cpu').numpy()

                    input_queue.put((iteration, output_np, target_np))

                    elapsed_time = timer.toc()
                    util.print_progress(iteration, len(self.dataset), 'Test Progress ({}p - {}sec):'.format(lr, round(elapsed_time, 2)), 'Complete', 1, 50)

                #terminate
                for _ in range(len(process_list)):
                    input_queue.put(('end', ))

                #merge results
                quality_list = []
                for process in process_list:
                    quality_list.extend(output_queue.get())

                for quality in quality_list:
                    result[lr].frameidx.append(quality.idx)
                    result[lr].ssim.append(quality.ssim)
                    result[lr].psnr.append(quality.psnr)

                result[lr].frameidx, result[lr].ssim, result[lr].psnr = \
                    [list(x) for x in zip(*sorted(zip(result[lr].frameidx, result[lr].ssim,result[lr].psnr), key=lambda pair: pair[0]))]

            #add HR info
            result[self.opt.dash_hr] = Result()
            result[self.opt.dash_hr].psnr = 100
            result[self.opt.dash_hr].ssim = 1

            return result

    def evaluate_quality(self):
        #summary log
        log_name = datetime.now().strftime('result_quality_summary_{}.log'.format(opt.test_num_epoch))
        summary_logger = util.get_logger(opt.result_dir, log_name)
        log = ''
        log += 'outputIdx\t'
        for lr in opt.dash_lr:
            log += 'PSNR(SR, {}p)\t'.format(lr)
            log += 'PSNR(bicubic, {}p)\t'.format(lr)
            log += '\t'
            log += 'SSIM(SR, {}p)\t'.format(lr)
            log += 'SSIM(bicubic, {}p)\t'.format(lr)
            log += '\t'
        summary_logger.info(log)

        #detail log (per frame)
        detail_logger = {}
        for output_node in self.output_nodes:
            detaill_logname = datetime.now().strftime('result_quality_detail_{}_{}.log'.format(output_node, opt.test_num_epoch))
            detail_logger[output_node] = util.get_logger(opt.result_dir, detaill_logname)

        for output_node in self.output_nodes:
            log = ''
            log += 'FrameIdx\t'
            for lr in self.node2res[output_node]:
                log += 'PSNR(SR, {}p)\t'.format(lr)
                log += 'PSNR(bicubic, {}p)\t'.format(lr)
                log += '\t'
                log += 'SSIM(SR, {}p)\t'.format(lr)
                log += 'SSIM(bicubic, {}p)\t'.format(lr)
                log += '\t'
            detail_logger[output_node].info(log)

        #analyze
        baseline_result = self._analyze_baseline()
        sr_result = {}
        for output_node in self.output_nodes:
            sr_result[output_node] = self._analyze_sr(output_node)

        #logging
        for output_node in self.output_nodes:
            #analyze
            log = ''
            log += '{}\t'.format(output_node)
            for lr in opt.dash_lr:
                if lr in self.node2res[output_node]:
                    log += '{}\t'.format(np.mean(sr_result[output_node][lr].psnr))
                    log += '{}\t'.format(np.mean(baseline_result[lr].psnr))
                    log += '\t'
                    log += '{}\t'.format(np.mean(sr_result[output_node][lr].ssim))
                    log += '{}\t'.format(np.mean(baseline_result[lr].ssim))
                    log += '\t'
                else:
                    log += '\t'
                    log += '\t'
                    log += '\t'
                    log += '\t'
                    log += '\t'
                    log += '\t'
            summary_logger.info(log)

            for idx in range(len(self.dataset)):
                log = ''
                log += '{}\t'.format(idx)
                for lr in opt.dash_lr:
                    if lr in self.node2res[output_node]:
                        log += '{}\t'.format(sr_result[output_node][lr].psnr[idx])
                        log += '{}\t'.format(baseline_result[lr].psnr[idx])
                        log += '\t'
                        log += '{}\t'.format(sr_result[output_node][lr].ssim[idx])
                        log += '{}\t'.format(baseline_result[lr].ssim[idx])
                        log += '\t'
                    else:
                        log += '\t'
                        log += '\t'
                        log += '\t'
                        log += '\t'
                        log += '\t'
                        log += '\t'
                detail_logger[output_node].info(log)

    #measure dnn inference time
    def evaluate_runtime(self):
        log_name = datetime.now().strftime('result_runtime.log')
        summary_logger = util.get_logger(opt.result_dir, log_name)
        result = {}
        log = ''
        log += 'OutputNode\t'
        for lr in opt.dash_lr:
            log += '{}p\t'.format(lr)
            result[lr] = {}
        summary_logger.info(log)

        batch_num = self.opt.test_num_batch
        for lr in opt.dash_lr:
            self.dataset.setTargetLR(lr)
            self.model.setTargetScale(self.dataset.getTargetScale())
            for node in self.model.getOutputNodes():
                elapsed_times = []
                t_w = RESOLUTION[lr][0]
                t_h = RESOLUTION[lr][1]
                input = torch.FloatTensor(batch_num, 3, t_w, t_h).random_(0,1).to(self.device)

                try:
                    for _ in range(DUMMY_TRIAL):
                        output = self.model(input, node)
                        torch.cuda.synchronize()

                    for _ in range(TEST_TRIAL):
                        start_time = time.perf_counter()
                        output = self.model(input, node)
                        torch.cuda.synchronize()
                        end_time = time.perf_counter()
                        elapsed_time = (end_time - start_time)
                        elapsed_times.append(elapsed_time)

                except Exception as e:
                    print(e)
                    sys.exit()

                average_elapsed_time = np.sum(elapsed_times) / (TEST_TRIAL * batch_num)
                result[lr][node] = average_elapsed_time

                print('[Resolution: Size ({}x{}), OutputNode: {}] / Inference time per frame(sec) {} / Max-Min(sec) {}'.format(t_w, t_h, node, round(average_elapsed_time, 4), round(np.max(elapsed_times) - np.min(elapsed_times), 4)))

        for node in self.output_nodes:
            log = ''
            log += '{}\t'.format(node)
            for lr in self.node2res[node]:
                log += '{}\t'.format(round(result[lr][node],4))
            summary_logger.info(log)

if __name__ == "__main__":
    model = MultiNetwork(template.get_nas_config(opt.quality))
    dataset = DatasetForDASH(opt)
    evaluator = Tester(opt, model, dataset)
    #evaluator.evaluate_quality()
    evaluator.evaluate_runtime()
