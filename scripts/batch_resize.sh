#!/bin/bash

convert_files() {
	SOURCE="${1}"
	RES="${2}"
	
	echo Resizing images from ${SOURCE}
	for IMG in ${SOURCE}/*.jpg
	do
		echo $IMG
		eval convert -resize ${RES}x${RES}^ -gravity center -extent ${RES}x${RES} $IMG $IMG
	done
}

convert_files "$1" "$2"