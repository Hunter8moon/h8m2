#!/usr/bin/env bash

ffmpeg -i "$1" -vf fps=30 "$2"/%d.png