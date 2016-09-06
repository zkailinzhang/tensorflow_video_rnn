#!/bin/bash

# convert all the video file to images
for file in data/*/*/*.avi; do ffmpeg -i "$file" "${file%.avi}"-%05d.png; done