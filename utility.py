import logging, os, math, re, ntpath, sys, subprocess, datetime, time, shlex, json
import numpy as np
import torch
import torch.nn as nn

THREAD_NUM=4

def write_frame(imageloc, videoloc, t_w, t_h, interpolation, extract_fps):
    command = ['ffmpeg',
               '-loglevel', 'fatal',
               '-i', videoloc,
               '-threads', str(THREAD_NUM),
               '-vf', 'scale=%d:%d, fps=%f'%(t_w, t_h, extract_fps),
               '-sws_flags', '%s'%(interpolation),
               '{}/%d.png'.format(imageloc)]

    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE ,stdout = subprocess.PIPE )
    out, err = ffmpeg.communicate()
    if(err) : print('error',err); return None;

def write_frame_noscale(imageloc, videoloc, extract_fps):
    command = ['ffmpeg',
               '-loglevel', 'fatal',
               '-i', videoloc,
               '-threads', str(THREAD_NUM),
               '-vf', 'fps=%f'%(extract_fps),
               '{}/%d.png'.format(imageloc)]

    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE ,stdout = subprocess.PIPE )
    out, err = ffmpeg.communicate()
    if(err) : print('error',err); return None;

def get_video_info(fileloc) :
    command = ['ffprobe',
               '-v', 'fatal',
               '-show_entries', 'stream=width,height,r_frame_rate,duration',
               '-of', 'default=noprint_wrappers=1:nokey=1',
               fileloc, '-sexagesimal']
    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE ,stdout = subprocess.PIPE )
    out, err = ffmpeg.communicate()
    if(err) : print(err)
    out = out.decode().split('\n')

    cmd = "ffprobe -v quiet -print_format json -show_streams"
    args = shlex.split(cmd)
    args.append(fileloc)
    ffprobeOutput = subprocess.check_output(args).decode('utf-8')
    ffprobeOutput = json.loads(ffprobeOutput)
    bitrate = int(ffprobeOutput['streams'][0]['bit_rate']) / 1000

    return {'file' : fileloc,
            'width': int(out[0]),
            'height' : int(out[1]),
            'fps': float(out[2].split('/')[0])/float(out[2].split('/')[1]),
            'duration' : out[3],
            'bitrate': bitrate}

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

class videoInfo:
    def __init__(self, fps, duration, quality):
        self.fps = fps
        self.duration = duration
        self.quality = quality

class timer():
    def __init__(self):
        self.acc = 0
        self.total_time = 0
        self.tic()

    def tic(self):
        self.t0 = time.perf_counter()

    def toc(self):
        return time.perf_counter() - self.t0

    def toc_total_sum(self):
        elapsed_time = time.perf_counter() - self.t0
        return self.total_time + elapsed_time

    def toc_total_add(self):
        elapsed_time = time.perf_counter() - self.t0
        self.total_time += elapsed_time
        return elapsed_time

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

    def add_total(time):
        self.total_time += time

def get_psnr(pred, gt, max_value=1.0):
    mse = np.mean((pred-gt)**2)
    if mse == 0:
        return 100
    else:
        return 20*math.log10(max_value/math.sqrt(mse))

def print_progress(iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def get_logger(save_dir, save_name):
    Logger = logging.getLogger(save_name)
    Logger.setLevel(logging.INFO)
    Logger.propagate = False

    filePath = os.path.join(save_dir, save_name)
    if os.path.exists(filePath):
        os.remove(filePath)

    fileHandler = logging.FileHandler(filePath)
    logFormatter = logging.Formatter('%(message)s')
    fileHandler.setFormatter(logFormatter)
    Logger.addHandler(fileHandler)

    return Logger
