#!/usr/bin/env bash

ffmpeg -ss 00:05:00.0 -i "$1" -c copy -t 00:03:00.0 "$2"