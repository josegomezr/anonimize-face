#!/bin/bash

INPUT_FILE="$1"

if [[ ! -f "$INPUT_FILE" ]]; then
	echo "NOT FOUND: $INPUT_FILE"
	exit 1
fi

BASENAME="${INPUT_FILE%.*}"
OVERLAY_FILE="${BASENAME}.overlay.mp4" # TODO: make this smarter somehow

if [[ ! -f "$OVERLAY_FILE" ]]; then
	echo "NOT FOUND: $OVERLAY_FILE"
	exit 1
fi


ffmpeg -i "$INPUT_FILE" -i "$OVERLAY_FILE" \
  -filter_complex "
	[1]split[m][a];
  	[a]geq='if(gt(lum(X,Y),16),255,0)',hue=s=0[al];
  	[m][al]alphamerge[3];
  	[0][3]overlay[4];
  	[4]gblur=sigma=5" \
  "${BASENAME}.merged.mp4"