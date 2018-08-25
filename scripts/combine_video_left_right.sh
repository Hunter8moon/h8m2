#!/usr/bin/env bash

ffmpeg \
  -i "$1" \
  -i "$2" \
  -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' \
  -map [vid] \
  -c:v libx264 \
  -crf 23 \
  -preset fast \
  "$3"