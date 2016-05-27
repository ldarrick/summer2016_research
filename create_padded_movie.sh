#!/bin/bash
#
# Author: Darrick Lee <y.l.darrick@gmail.com>
#
# This script first pads all images in the current folder with the given color so that
# it is at the correct resolution, and also converts it to jpg. According to the link below,
# mencoder gives the highest quality videos with jpg format.
#
# Reference: http://electron.mit.edu/~gsteele/ffmpeg/

COLOR=$1
WIDTH=$2
HEIGHT=$3
FPS=$4
FORMAT=$5
OUTPUT=$6
BITRATE=$7

DIM="${WIDTH}x${HEIGHT}"

if ! [ -d paddedImages/ ]; then
	mkdir paddedImages/
fi

for im in $(ls *.png); do
	echo $im
	imjpg="$(basename "$im" .png).jpg"
	convert ${im} -gravity center -background $COLOR -extent $DIM "paddedImages/${imjpg}"
done

cd paddedImages/

mencoder mf://*.jpg -mf w=${WIDTH}:h=${HEIGHT}:fps=${FPS}:type=jpg -ovc lavc \
	-lavcopts vcodec=msmpeg4v2:vbitrate=${BITRATE} -oac copy -fps ${FPS} -o ${OUTPUT}

exit 0
