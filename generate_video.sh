#!/bin/bash

OPTIND=1

while getopts ":i:f:r:" opt; do
    case "$opt" in
    i)
        input=$OPTARG
        ;;
    f)
        fps=$OPTARG
        ;;
    r)
        resolution=$OPTARG
        ;;
    *)
        echo "Usage: $0 -i [input_file] -f [fps] -r [resolution]" 1>&2; exit 1;
        ;;
    esac
done

if [ $OPTIND -eq 1 ]; then echo "Usage: $0 -i [input_file] -f [fps] -r [resolution]" 1>&2; exit 1; fi

echo "Extract & Generate {SR, LR} Video Frames using NAS-content-aw trained model"
python gen_video.py --quality ultra --data_name $input --use_cuda --load_on_memory --dash_lr $resolution --fps $fps

echo "Encode {SR, LR} Video with Generated Video Frames using FFMPEG"
ffmpeg -r ${fps} -i ./result/${input}/ultra/${resolution}_%d_output.png -vcodec libx264 -preset medium -pix_fmt yuv420p ${input}_SR.mp4
ffmpeg -r ${fps} -i ./result/${input}/ultra/${resolution}_%d_input.png -vcodec libx264 -preset medium -pix_fmt yuv420p ${input}_LR.mp4
ffmpeg -r ${fps} -i ./result/${input}/ultra/${resolution}_%d_target.png -vcodec libx264 -preset medium -pix_fmt yuv420p ${input}_HR.mp4
