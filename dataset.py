import shutil, glob, random, time, os, ntpath, sys, re, logging, math, random
import numpy as np
from PIL import Image, ImageFile
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor, Resize

import utility as util
from option import opt
from scipy import misc

"""
Dataset for DASH content which provides (lr, hr) images for multiple resolutions

opt: argument is listed in option.py
"""
class DatasetForDASH(data.Dataset):
    def __init__(self, opt):
        super(DatasetForDASH, self).__init__()
        assert os.path.exists(opt.data_dir)
        self.data_dir = opt.data_dir
        self.fps = opt.fps
        self.hr = opt.dash_hr
        self.lr = opt.dash_lr
        self.interpolation = opt.interpolation
        self.patch_size = opt.patch_size
        self.num_batch = opt.num_batch
        self.num_update_per_epoch = opt.num_update_per_epoch
        self.num_valid_image = opt.num_valid_image
        self.load_on_memory = opt.load_on_memory

        self.input_transform = Compose([ToTensor(),])
        self.target_tarnsform = Compose([ToTensor(),])
        self.dataset_type = None #choose between train, valid, test
        self.target_lr = None #choose between low-resolutions

        self._setup()

    def _getPathInfo(self, resolution, upscale):
        checkfile = os.path.join(self.data_dir, 'prepare-{}p-{}fps'.format(resolution, self.fps))
        frames_dir = os.path.join(self.data_dir, '{}p-{}fps'.format(resolution, self.fps))
#print('here !! ', frames_dir)
        if not upscale and resolution != self.hr:
            checkfile += '-no-upscale'
            frames_dir += '-no-upscale'

        return checkfile, frames_dir

    def _setup(self):
        #Get video information
        self.videos = {}
        for lr in self.lr:
            lr_dir = os.path.join(self.data_dir, '{}p'.format(lr))
            filenames = glob.glob('{}/output*k.mp4'.format(lr_dir))
            assert len(filenames) == 1
            self.videos[lr] = util.get_video_info(filenames[0])

        hr_dir = os.path.join(self.data_dir, '{}p'.format(self.hr))
        filenames = glob.glob('{}/output*k.mp4'.format(hr_dir))
        assert len(filenames) == 1
        self.videos[self.hr] = util.get_video_info(filenames[0])

        #Configure input, target resolution for super-resolution neural networks
        for lr in self.lr:
            self.videos[lr]['scale'] = self.videos[self.hr]['width'] // self.videos[lr]['width']
            self.videos[lr]['input_width'] = self.videos[self.hr]['width'] / self.videos[lr]['scale']
            self.videos[lr]['input_height'] = self.videos[self.hr]['height'] / self.videos[lr]['scale']
            self.videos[lr]['target_width'] = self.videos[self.hr]['width']
            self.videos[lr]['target_height'] = self.videos[self.hr]['height']

            assert self.videos[self.hr]['width'] % self.videos[lr]['scale'] == 0
            assert self.videos[self.hr]['height'] % self.videos[lr]['scale'] == 0

        #Prepare image dataset extracted from videos
        self.lr_upscaled_frames_dir = []
        self.lr_frames_dir = []
        self.hr_frames_dir = None

        for resolution, video in self.videos.items():
            print('===>Prepare {}p images'.format(resolution))
            if resolution != self.hr: #low-resolution
                checkfile, frames_dir = self._getPathInfo(video['height'], True)
                if not os.path.exists(checkfile):
                    if os.path.exists(frames_dir) and os.path.isdir(frames_dir):
                        shutil.rmtree(frames_dir)
                    os.makedirs(frames_dir)
                    util.write_frame(frames_dir, video['file'], video['target_width'], video['target_height'], self.interpolation, self.fps)
                    with open(checkfile, 'w+') as f:
                        f.write('done')
                self.lr_upscaled_frames_dir.append(frames_dir)

                checkfile, frames_dir = self._getPathInfo(video['height'], False)
                #self.lr_frames_dir.append(frames_dir)
#print ('append', frames_dir)
                if not os.path.exists(checkfile):
                    if os.path.exists(frames_dir) and os.path.isdir(frames_dir):
                        shutil.rmtree(frames_dir)
                    os.makedirs(frames_dir)
                    util.write_frame(frames_dir, video['file'], video['input_width'], video['input_height'], self.interpolation, self.fps)
                    with open(checkfile, 'w+') as f:
                        f.write('done')
                self.lr_frames_dir.append(frames_dir)
            else: #high-resolution
                checkfile, frames_dir = self._getPathInfo(video['height'], True)
                if not os.path.exists(checkfile):
                    if os.path.exists(frames_dir) and os.path.isdir(frames_dir):
                        shutil.rmtree(frames_dir)
                    os.makedirs(frames_dir)
                    util.write_frame_noscale(frames_dir, video['file'], self.fps)
                    with open(checkfile, 'w+') as f:
                        f.write('done')
                self.hr_frames_dir = frames_dir

        #Read images
        filenames = glob.glob('{}/*.png'.format(self.hr_frames_dir))
        filenames.sort(key=util.natural_keys)
        self.hr_filenames = filenames
        self.lr_upscaled_filenames = []
        for frames_dir in self.lr_upscaled_frames_dir:
            filenames = glob.glob('{}/*.png'.format(frames_dir))
            assert len(filenames) == len(self.hr_filenames)
            filenames.sort(key=util.natural_keys)
            self.lr_upscaled_filenames.append(filenames)
        self.lr_filenames = []
        for frames_dir in self.lr_frames_dir:
            filenames = glob.glob('{}/*.png'.format(frames_dir))
            assert len(filenames) == len(self.hr_filenames)
            filenames.sort(key=util.natural_keys)
            self.lr_filenames.append(filenames)

#print (self.lr_filenames[self.lr.index(720)])
#print (len(self.lr_filenames))
#       print (self.lr.index(720))
#       print (self.lr_frames_dir)
        #(Optional) Load images on memory
        if self.load_on_memory:
            print('===>Load images on memory')
            if self.dataset_type is not 'train':
                self.lr_upscaled_images = []
                for input_filenames in self.lr_upscaled_filenames:
                    img_list = []
                    for name in input_filenames:
                        img = Image.open(name)
                        img.load()
                        img_list.append(img)
                    self.lr_upscaled_images.append(img_list)

            self.lr_images = []
            for input_filenames in self.lr_filenames:
                img_list = []
                for name in input_filenames:
#print(name)
                    img = Image.open(name)
                    img.load()
                    img_list.append(img)
                self.lr_images.append(img_list)

            self.hr_images = []
            for name in self.hr_filenames:
                img = Image.open(name)
                img.load()
                self.hr_images.append(img)

    def setDatasetType(self, dataset_type):
        assert dataset_type in ['train', 'valid', 'test']
        self.dataset_type = dataset_type

    def setTargetLR(self, lr):
        assert lr in self.lr
        self.target_lr = lr

    def getTargetScale(self):
        return self.videos[self.target_lr]['scale']

    def getBitrate(self, resolution):
        return self.videos[resolution]['bitrate']

    def getItemTrain(self):
        frame_idx = random.randrange(0, len(self.hr_filenames))

        if self.load_on_memory:
            input = self.lr_images[self.lr.index(self.target_lr)][frame_idx]
#input_test = self.input_transform(input)
            #assert (self.videos[self.target_lr]['width'] in input_test.shape)
            target = self.hr_images[frame_idx]
        else:
#if (self.target_lr == 360):
#               print (self.lr_filenames[self.lr.index(self.target_lr)][frame_idx])
            input = Image.open(self.lr_filenames[self.lr.index(self.target_lr)][frame_idx])
            input.load()
            target = Image.open(self.hr_filenames[frame_idx])
            target.load()

        #Randomly select crop lcation
        width, height = input.size
        #print (width, height)

        scale = self.videos[self.target_lr]['scale']
        height_ = random.randrange(0, height - self.patch_size + 1)
        width_ = random.randrange(0, width - self.patch_size + 1)
        input_ori = input 
        assert (height_ + self.patch_size <= height)
        assert (width_ + self.patch_size <= width)
        input = input.crop((width_ , height_, width_ + self.patch_size, height_ + self.patch_size))
        target = target.crop((width_ * scale, height_ * scale, (width_ + self.patch_size) * scale, (height_ + self.patch_size) * scale))
#misc.imsave('{}_{}_w{}_h{}_input.png'.format(self.target_lr, frame_idx, width_, height_), input)
#misc.imsave('{}_{}_w{}_h{}_target.png'.format(self.target_lr, frame_idx, width_, height_), target)
#misc.imsave('{}_{}_w{}_h{}_input_ori.png'.format(self.target_lr, frame_idx, width_, height_), input_ori)

        input = self.input_transform(input)
        target = self.input_transform(target)
        #print (input.shape)
        return input, target

    def getItemTest(self, index):
        if self.load_on_memory:
#print (self.lr.index(self.target_lr))
            input = self.lr_images[self.lr.index(self.target_lr)][index]
            target = self.hr_images[index]
            upscaled = self.lr_upscaled_images[self.lr.index(self.target_lr)][index]
        else:
#print (self.lr_filenames[self.lr.index(self.target_lr)][index])
 
            input = Image.open(self.lr_filenames[self.lr.index(self.target_lr)][index])
            input.load()
            target = Image.open(self.hr_filenames[index])
            target.load()
            upscaled = Image.open(self.lr_upscaled_filenames[self.lr.index(self.target_lr)][index])
            upscaled.load()

        input = self.input_transform(input)
#print (input.shape)
        upscaled = self.input_transform(upscaled)
        target = self.input_transform(target)

        return input, upscaled, target

    def lenTrain(self):
        return self.num_batch * self.num_update_per_epoch

    def lenValid(self):
        return self.num_valid_image

    def lenTest(self):
        return len(self.hr_filenames)

    def __getitem__(self, index):
        if self.dataset_type == 'train':
            return self.getItemTrain()
        elif self.dataset_type == 'valid':
            return self.getItemTest((self.lenTest()//self.num_valid_image) * index)
        elif self.dataset_type == 'test':
            return self.getItemTest(index)
        else:
            raise NotImplementedError

    def __len__(self):
        if self.dataset_type == 'train':
            return self.lenTrain()
        elif self.dataset_type == 'valid':
            return self.lenValid()
        elif self.dataset_type == 'test':
            return self.lenTest()
        else:
            raise NotImplementedError

if __name__ == "__main__":
    """Simple test code for dataset"""
    dataset = DatasetForDASH(opt)
    dataset.setDatasetType('train')
    dataset.setTargetLR(240)
    print(len(dataset))
    train_dataloader = data.DataLoader(dataset=dataset, num_workers=opt.num_thread, batch_size=opt.num_batch, pin_memory=True, shuffle=True)
    for iteration, batch in enumerate(train_dataloader):
        print(batch[0].size(), batch[1].size())
        break

    dataset.setDatasetType('valid')
    print(len(dataset))
    valid_dataloader = data.DataLoader(dataset=dataset, num_workers=opt.num_thread, batch_size=1, pin_memory=True, shuffle=True)
    for iteration, batch in enumerate(valid_dataloader):
        print(batch[0].size(), batch[1].size(), batch[2].size())
        break

"""
#Data-augmentation is enalbed by randomly selecting among 8 choices
class DatasetForDIV2K(data.Dataset):
    @staticmethod
    def input_transform():
        return Compose([
            ToTensor(),
        ])

    @staticmethod
    def target_transform():
        return Compose([
            ToTensor(),
        ])

    def __init__(self, opt, image_dir):
        super(DatasetForDIV2K, self).__init__()

        self.scale = opt.div2kLR
        self.patch_size = opt.patchSize
        self.batch_size = opt.batchSize
        self.image_dir = image_dir
        self.lr = [240, 360, 480, 720]
        self.target_lr = self.lr[0] #TODO: remove this defualt value
        self.interpolation = opt.resize

        self.input_transform = self.input_transform()
        self.target_transform = self.target_transform()

        #self.target_dir = os.path.join(self.image_dir, 'DIV2K_train_HR_modcrop{}.')
        self.target_dir= os.path.join(self.image_dir, 'train-1080p')
        print(self.target_dir)
        assert os.path.exists(self.target_dir)

        #self.target_filenames = [os.path.join(self.target_dir, x) for x in os.listdir(self.target_dir) if is_image_file(x)]
        self.target_filenames = glob.glob('{}/*.png'.format(self.target_dir))
        self.target_filenames.sort(key=util.natural_keys)

        #dataset creat for DIV2K image
        self.input_dirs = []
        for scale in self.scale:
            checkfile = os.path.join(self.image_dir, 'prepare-train-{}-{}'.format(scale, self.interpolation))
            upscaled_dir = os.path.join(image_dir, 'lr_x{}_{}'.format(scale, self.interpolation))

            if not os.path.exists(checkfile):
                logging.info('===>Prepare x{} images'.format(scale))
                if os.path.exists(upscaled_dir) and os.path.isdir(upscaled_dir):
                    shutil.rmtree(upscaled_dir)
                os.makedirs(upscaled_dir)

                for path in self.target_filenames:
                    util.resize_div2k(upscaled_dir, path, scale, self.interpolation, 1, True)

                with open(checkfile, 'w+') as f:
                    f.write('div2k dataset is prepared')

            self.input_dirs.append(upscaled_dir)

        #Make a filenames list and sort it
        self.input_filenames = []
        for input_dir in self.input_dirs:
            #input_filenames = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if is_image_file(x)]
            input_filenames = glob.glob('{}/*.png'.format(input_dir))
            input_filenames.sort(key=util.natural_keys)
            self.input_filenames.append(input_filenames)

        #Lazy loading rather than load on memory
        #Load an image on memory
        print('===> Load input images on memory')
        self.input_images = []
        for input_filenames in self.input_filenames:
            img_list = []
            for name in input_filenames:
                img = Image.open(name)
                img.load()
                img_list.append(img)
            self.input_images.append(img_list)

        print('===> Load target images on memory')
        self.target_images = []
        for name in self.target_filenames:
            img = Image.open(name)
            img.load()
            self.target_images.append(img)

    def __getitem__(self, index):
        #input_index = np.random.randint(0, len(self.input_filenames))
        image_index = random.randrange(0, len(self.target_filenames))

        #get an image from memory
        #input = Image.open(self.input_filenames[input_index][image_index])
        #target = Image.open(self.target_filenames[image_index])

        input = self.input_images[self.lr.index(self.target_lr)][image_index]
        target = self.target_images[image_index]

        #crop
        height, width = target.size
        scale = self.scale[self.lr.index(self.target_lr)]
        height_ = random.randrange(0, height - self.patch_size + 1)
        width_ = random.randrange(0, width - self.patch_size + 1)

        input = input.crop((width_, height_, width_ + self.patch_size, height_ + self.patch_size))
        target = target.crop((width_ * scale, height_ * scale, (width_ + self.patch_size) * scale, (height_ + self.patch_size) * scale))

        #print(input.size, target.size)

        #input = self.input_transform(input)
        #upscaled = self.input_transform(upscaled)
        #target = self.input_transform(target)

        #transform
        input = self.input_transform(input)
        target = self.target_transform(target)

        return input, target

    def getScaleList(self):
        return self.scale

    def setScaleIdx(self, idx_scale):
        assert idx_scale < len(self.scale)
        self.target_lr = self.lr[idx_scale]

    def setScale(self, scale):
        #print(self.scale)
        assert scale in self.scale
        self.target_lr = self.lr[self.scale.index(scale)]

    def setTargetLR(self, lr):
        self.target_lr = lr

    def __len__(self):
        #return self.batch_size * 100
        return self.batch_size * 1000
        #return self.batch_size * 10000 #iterate 1000 mini-batches per epoch

#Data-augmentation is enalbed by randomly selecting among 8 choices
class TestDatasetForDIV2K(data.Dataset):
    @staticmethod
    def input_transform():
        return Compose([
            ToTensor(),
        ])

    @staticmethod
    def target_transform():
        return Compose([
            ToTensor(),
        ])

    def __init__(self, image_dir, scale, interpolation, image_format):
        super(TestDatasetForDIV2K, self).__init__()
        self.scale = scale
        self.current_scale = scale[0]
        self.image_dir = image_dir
        self.image_format = image_format
        self.interpolation = interpolation

        self.input_transform = self.input_transform()
        self.target_transform = self.target_transform()

        self.target_dir = os.path.join(image_dir, 'DIV2K_valid_HR')
        assert os.path.exists(self.target_dir)

        self.target_filenames = [os.path.join(self.target_dir, x) for x in os.listdir(self.target_dir) if is_image_file(x)]
        self.target_filenames.sort(key=natural_keys)

        #dataset creat for DIV2K image
        self.input_dirs = {}
        for scale in self.scale:
            checkfile = os.path.join(self.image_dir, 'prepare-valid-{}-{}-{}'.format(scale, self.image_format, self.interpolation))
            upscaled_dir = os.path.join(image_dir, 'lr_x{}_upscaled_{}_{}_valid'.format(scale, self.image_format, self.interpolation))

            if not os.path.exists(checkfile):
                logging.info('===>Prepare x{} images'.format(scale))
                if os.path.exists(upscaled_dir) and os.path.isdir(upscaled_dir):
                    shutil.rmtree(upscaled_dir)
                os.makedirs(upscaled_dir)

                for path in self.target_filenames:
                    util.resize_div2k(upscaled_dir, path, scale, self.interpolation, self.image_format)

                with open(checkfile, 'w+') as f:
                    f.write('div2k dataset is prepared')

            self.input_dirs[scale] = upscaled_dir

        #Make a filenames list and sort it
        self.input_filenames = {}
        for scale, input_dir in self.input_dirs.items():
            input_filenames = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if is_image_file(x)]
            input_filenames.sort(key=natural_keys)
            self.input_filenames[scale] = input_filenames

        #Lazy loading rather than load on memory
        #Load an image on memory
        #print('===> Load input images on memory')
        self.input_images = {}
        for scale, input_filenames in self.input_filenames.items():
            img_list = []
            for name in input_filenames:
                img = Image.open(name)
                img.load()
                img_list.append(img)
            self.input_images[scale] = img_list

        #print('===> Load target images on memory')
        self.target_images = []
        for name in self.target_filenames:
            img = Image.open(name)
            img.load()
            self.target_images.append(img)

    def setscale(self, scale):
        self.current_scale = scale

    def __getitem__(self, index):
        input = self.input_images[self.current_scale][index]
        target = self.target_images[index]

        #transform
        input = self.input_transform(input)
        target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.target_images)
"""
