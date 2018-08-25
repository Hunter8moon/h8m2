#!/usr/bin/env bash

ffmpeg -pattern_type glob -i '*.png' -c:v libx264  "$1"