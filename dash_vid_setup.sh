#!/bin/bash

function x264_encode() {
    local bitrate=$1
    local width=$2
    local height=$3
    local input_video=$4
    local method=$5

    # to make similar video segment sizes => give --vbv-maxrate, --vbv-bufsize option same bitrate as --bitrate option
    # recommended : --bitrate = bitrate --vbv-maxrate = bitrate * 2, --vbv-bufsize = bitrate * 4

    x264 --output intermediate_${bitrate}k.264 --fps 24 --preset slow --bitrate ${bitrate} --vbv-maxrate $((bitrate * 2)) --vbv-bufsize $((bitrate * 4)) --min-keyint 96 --keyint 96 --scenecut 0 --no-scenecut --pass 1 --video-filter "resize:width=${width},height=${height},method=${method}" ${input_video}
    #x264 --output intermediate_${bitrate}k.264 --preset slow --bitrate ${bitrate} --vbv-maxrate ${bitrate} --vbv-bufsize ${bitrate} --min-keyint 96 --keyint 96 --scenecut 0 --no-scenecut --pass 1 --video-filter "resize:width=${width},height=${height},method=${method}" ${input_video}
}

function MP4Box_MP4() {
    local bitrate=$1
    MP4Box -add intermediate_${bitrate}k.264 -fps 24 output_${bitrate}k.mp4
}

function MP4Box_Dash() {
    local bitrate=$1
    local dir=$2
    cd ./$dir
    MP4Box -dash 4000 -frag 4000 -rap -segment-name segment_ ../../output_${bitrate}k.mp4
    cd ../
}

function MP4Box_Dash_singleMPD() {
    local bitrate1=$1
    local bitrate2=$2
    local bitrate3=$3
    local bitrate4=$4
    local bitrate5=$5

    MP4Box -dash 4000 -frag 4000 -rap -profile dashavc264:live -bs-switching no -url-template output_${bitrate1}k.mp4:id="1080p" output_${bitrate2}k.mp4:id="720p" output_${bitrate3}k.mp4:id="480p" output_${bitrate4}k.mp4:id="360p" output_${bitrate5}k.mp4:id="240p" -segment-name '$RepresentationID$/segment_$Number$$Init=init' -out multi_resolution.mpd

    mv ./output_${bitrate1}* 1080p
    mv ./output_${bitrate2}* 720p
    mv ./output_${bitrate3}* 480p
    mv ./output_${bitrate4}* 360p
    mv ./output_${bitrate5}* 240p
}

echo "Dash Content Script"

OPTIND=1

while getopts ":i:" opt; do
    case "$opt" in
    i)
        input=$OPTARG
        ;;
    *)
        echo "Usage: $0 -i [input_file]" 1>&2; exit 1;
        ;;
    esac
done

if [ $OPTIND -eq 1 ]; then echo "Usage: $0 -i [input_file]" 1>&2; exit 1; fi


#command -v x264 >/dev/null 2>&1 || { echo >&2 "x264 not installed"; exit 1;}
#command -v MP4 >/dev/null 2>&1 || { echo >&2 "MP4 not installed"; exit 1;}


echo "x264 encoding"
x264_encode "4800" "1920" "1080" $input bicubic
x264_encode "2400" "1280" "720" $input bicubic
x264_encode "1200" "854" "480" $input bicubic
x264_encode "800" "640" "360" $input bicubic
x264_encode "400" "426" "240" $input bicubic

echo "encoding to MP4"
MP4Box_MP4 "4800"
MP4Box_MP4 "2400"
MP4Box_MP4 "1200"
MP4Box_MP4 "800"
MP4Box_MP4 "400"

base_dir=$(basename $input .mp4)

echo "mkdir input dir"
mkdir -p ./1080p
mkdir -p ./720p
mkdir -p ./480p
mkdir -p ./360p
mkdir -p ./240p

MP4Box_Dash_singleMPD "4800" "2400" "1200" "800" "400"
