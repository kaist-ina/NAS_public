import os, sys, logging, math, random, time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import imageio

import utility as util

class Trainer():
    def __init__(self, opt, model, dataset):
        self.opt = opt
        self.model = model
        self.dataset = dataset
        self.device = torch.device("cuda" if opt.use_cuda else "cpu")
        self.timer = util.timer()
        self.epoch = 0

        #build optimizer
        self.optimizer_dict = {}
        for scale, _ in model.scale_dict.items():
            self.optimizer_dict[scale] = optim.Adam(model.networks[model.scale_dict[scale]].parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        self.loss_func = self._get_loss_func(opt.loss_type)

        #load a model on a target device
        self.model = self.model.to(self.device)

    def _get_loss_func(self, loss_type):
        if loss_type == 'l2':
            return nn.MSELoss()
        elif loss_type == 'l1':
            return nn.L1Loss()
        else:
            raise NotImplementedError

    def _adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
        if self.opt.lr_decay_epoch is not None:
            lr = self.opt.lr * (self.opt.lr_decay_rate ** (epoch // self.opt.lr_decay_epoch))
        else:
            lr = self.opt.lr

        for _, optimizer in self.optimizer_dict.items():
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    def train_one_epoch(self):
        self.timer.tic()

        #decay learning rate
        self._adjust_learning_rate(self.epoch)

        #iterate over low-resolutions
        self.model.train()
        self.dataset.setDatasetType('train')
        train_dataloader = DataLoader(dataset=self.dataset, num_workers=self.opt.num_thread, batch_size=self.opt.num_batch, pin_memory=True, shuffle=True)

        for lr in self.opt.dash_lr:
            self.dataset.setTargetLR(lr)
            scale = self.dataset.getTargetScale()
            self.model.setTargetScale(scale)

            #iterate over training image patches
            for iteration, batch in enumerate(train_dataloader, 1):
                input, target = batch[0], batch[1]
                input, target =  input.to(self.device), target.to(self.device)
                
                self.optimizer_dict[scale].zero_grad()
                loss = self.loss_func(self.model(input), target)
                
                loss.backward()
                self.optimizer_dict[scale].step()

                if iteration % 10 == 0:
                    util.print_progress(iteration, len(self.dataset)/self.opt.num_batch, 'Train Progress ({}p):'.format(lr), 'Complete', 1, 50)

        self.epoch += 1
        print('Epoch[{}-train](complete): {}sec'.format(self.epoch, self.timer.toc()))

    def validate(self):
        with torch.no_grad():
            self.model.eval()
            self.dataset.setDatasetType('valid')
            valid_dataloader = DataLoader(dataset=self.dataset, num_workers=self.opt.num_thread, batch_size=1, pin_memory=True, shuffle=False)

            #iterate over low-resolutions
            for lr in self.opt.dash_lr:
                self.dataset.setTargetLR(lr)
                self.model.setTargetScale(self.dataset.getTargetScale())

                #iterate over validation images
                total_sr_psnr = {}
                total_baseline_psnr = []
                for iteration, batch in enumerate(valid_dataloader, 1):
                    input, upscaled, target = batch[0], batch[1], batch[2]
                    upscaled_np, target_np = torch.squeeze(upscaled, 0).permute(1, 2, 0).numpy(), torch.squeeze(target, 0).permute(1, 2, 0).numpy()
                    input, upscaled, target =  input.to(self.device), upscaled.to(self.device), target.to(self.device)

                    psnr_baseline = util.get_psnr(upscaled_np, target_np)
                    total_baseline_psnr.append(psnr_baseline)
 
                    #iterate over output nodes
                    for node in self.model.getOutputNodes():
                        output = self.model(input, node)
                        output = torch.squeeze(torch.clamp(output, min=0, max=1.), 0).permute(1, 2, 0)
                        output_np = output.to('cpu').numpy()
                        psnr_sr = util.get_psnr(output_np, target_np)

                        if node not in total_sr_psnr:
                            total_sr_psnr[node] = []
                            total_sr_psnr[node].append(psnr_sr)
                        else:
                            total_sr_psnr[node].append(psnr_sr)

                        #save an image for the last node
                        if node == self.model.getOutputNodes()[-1]:
                            output_np *= 255
                            upscaled_np *= 255
                            target_np *= 255                            
                            
                            imageio.imwrite('{}/{}_{}_output.png'.format(self.opt.result_dir, lr, iteration), output_np.astype(np.uint8))
                            imageio.imwrite('{}/{}_{}_baseline.png'.format(self.opt.result_dir, lr, iteration), upscaled_np.astype(np.uint8))
                            imageio.imwrite('{}/{}_{}_target.png'.format(self.opt.result_dir, lr, iteration), target_np.astype(np.uint8))

                    util.print_progress(iteration, len(self.dataset), 'Valid Progress ({}p):'.format(lr), 'Complete', 1, 50)

                for node in self.model.getOutputNodes():
                    print("Epoch[{}-validation-{}-{}p] PSNR (output): {:.3f} PSNR (baseline): {:.3f}".format(self.epoch, node, lr, np.mean(total_sr_psnr[node]), np.mean(total_baseline_psnr)))

    def save_model(self):
        save_path = os.path.join(self.opt.checkpoint_dir, 'epoch_{}.pth'.format(self.epoch))
        torch.save(self.model.state_dict(), save_path)

    def save_dnn_chunk(self):
        self.model.save_chunk(self.opt.checkpoint_dir)

if __name__ == "__main__":
    train()
