# NAS
This repository is an official implementation of the paper [Neural Adaptive Content-aware Internet Video Delivery](https://ina.kaist.ac.kr/~nas).

Currently, we only provide NAS-MDSR, which is a super-resolution module of NAS.

### Prerequisites (p: Python package, b: Binary)

* Python 3.6
* (p) PyTorch >= 1.0.0
* (p) numpy
* (p) skimage
* (p) scipy
* (p) cv2 **(Use opencv package from [here](https://anaconda.org/conda-forge/opencv))**
* (p) pillow
* (p) ffmpeg
* (b) MP4Box, x264 (Refer [here](https://bitmovin.com/mp4box-dash-content-generation-x264/) for installing these binaries)

### Prepare MPEG-DASH dataset 

Download MPEG-DASH dataset from [here](https://www.dropbox.com/sh/tmfvbikh3gad7qy/AAAeptwDpHHg9FkVGaNAMV34a?dl=0) and place like:
```
data
 |------news
        |------original.mp4
        |------240p
        |------360p
        |------480p
        |------720p
        |------1080p
```

Or, if you want to use own 1080p video, first place it like:
```
data
 |------[dataset name]
        |------[video file]
```

Then, to generate MPEG-DASH content from the original video:
```
cd data/[dataset name]
../../dash_vid_setup.sh -i [video_file]
```

* Original video link for provided news dataset: [here](https://www.youtube.com/watch?v=4AtOU0dDXv8)

### How to train NAS-MDSR

To train NAS-MDSR: 
```
python train_nas_awdnn.py --quality [quality level] --data_name [dataset name] --use_cuda --load_on_memory
```
NAS-MDSR provides total four quality levels as in the paper (e.g., low, medium, high, ultra-high).
The higher the quality level, the bigger the model size, the higher the model quality.

Models will be saved like:
```
checkpoint
 |------[dataset name]
        |------[quality level]
                |------epoch_[index].pth
                |------      ...
                |------DNN_chunk_1.pth
                |------DNN_chunk_2.pth
                |------DNN_chunk_3.pth
                |------DNN_chunk_4.pth
                |------DNN_chunk_5.pth
```
DNN chunks are save only for the last updated model.
These chunks are used for streaming together with video chunks in NAS.

* Related code: model.py, dataset.py, option.py, trainer.py, train_nas_awdnn.py

### How to test NAS-MDSR (image)

To measure the quality of NAS-MDSR both in PSNR,SSIM:
```
python test_nas_quality.py --quality [quality level] --data_name [dataset name] --use_cuda --load_on_memory
```
Result will be saved like:
```
result
 |------[dataset name]
        |------[quality level]
                |------result_quality_detail_0_[epoch].log
                |------result_quality_detail_2_[epoch].log
                |------result_quality_detail_4_[epoch].log
                |------result_quality_detail_6_[epoch].log
                |------result_quality_detail_8_[epoch].log
                |------result_quality_summary_[epoch].log
```

To measure the inference time of NAS-MDSR:
```
python test_nas_runtime.py --quality [quality level] --data_name [dataset name] --use_cuda --load_on_memory
```
Result will be saved like:
```
result
 |------[dataset name]
        |------[quality level]
                |------result_runtime.log
```

* Related code: model.py, dataset.py, option.py, tester.py, test_nas_quality.py, test_nas_runtime.py

### How to test NAS-MDSR (video)

NAS-MDSR can also process a video in which decoding, encoding, super-resolution are done parallely.

To apply NAS-MDSR to process video chunks:
```
python test_nas_video_process.py --quality [quality level] --data_name [dataset name] --use_cuda --load_on_memory
```
It will generate quality-enhanced video chunks like:
```
result
 |------[dataset name]
            |------[quality level]
                |------[segment_[chunk index]_[resolution index]]
                        |------input.mp4
                        |------output.mp4 (quality-enhanced video chunk)
```
You can set chunk index and resolution index in test_nas_video_quality.py.

To measure the latency of NAS-MDSR to process video chunks:
```
python test_nas_video_runtime.py --quality [quality level] --data_name [dataset name] --use_cuda --load_on_memory
```
Result will be saved like:
```
result
 |------[dataset name]
            |------[quality level]
                    |------result_video_runtime.log
```
Refer process.py to understand detail procedures for processing video chunks.

* Related code: model.py, dataset.py, option.py, tester.py, test_nas_video_quality.py, test_nas_video_runtime.py

### Tip 

* Use the option 'load_on_memory' if you have enough memory since it highly affects on training speed.
* Use the option 'use_cuda' for using a GPU.

### Citation

If you find paper useful for your research, please cite our paper.

Hyunho, et al. "Neural adaptive content-aware internet video delivery." 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18). 2018. [[Website](http://ina.kaist.ac.kr/~nas/)] 
```
@inproceedings{yeo2018neural,
    title={Neural adaptive content-aware internet video delivery},
    author={Yeo, Hyunho and Jung, Youngmok and Kim, Jaehong and Shin, Jinwoo and Han, Dongsu},
    booktitle={13th $\{$USENIX$\}$ Symposium on Operating Systems Design and Implementation ($\{$OSDI$\}$ 18)},
    pages={645--661},
    year={2018}
}
```

### Author

Hyunho Yeo (PhD candidate at KAIST) / chaos5958@gmail.com
