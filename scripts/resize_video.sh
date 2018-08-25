#!/usr/bin/env bash

 ffmpeg -i "$1" -vf scale=-1:512 "$2"