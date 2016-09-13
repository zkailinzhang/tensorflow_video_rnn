#!/bin/bash

# convert all the video file to images
# the path/to/data structure will be something like /user/data/train_data/walk/video1.avi

for folder in /Volumes/dgu\'s\ passport/datasets/\(action\)LCA/video_data/validation_data/*
do
    count = 0
    for file in "$folder"/*.avi
    do
        if [[ ! -d "$folder"/$count ]]; then
            mkdir -p "$folder"/$count
        fi
        ffmpeg -i "$file" "$folder"/$count/%05d.png
        (( count++ ))
    done
done