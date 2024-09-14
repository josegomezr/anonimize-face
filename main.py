import logging
import numpy as np
import cv2
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s - %(name)s[%(processName)s|%(filename)s:%(lineno)d] %(message)s')

# patch numpy
setattr(np, 'int', int)

from video_processor import VideoProcessor

from stopwatch import Stopwatch
from utils import dump_state, inspect_video, draw_on_faces

def write_overlay(dest, vdata, bounding_boxes, context_window_size = 15):
    frame_size = vdata['frame_size']
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(dest), fourcc, vdata['fps'], frame_size)

    box_color=(0, 255, 0)

    pos = 0
    frame_count = len(bounding_boxes)
    
    half = context_window_size // 2

    logging.debug("Using {} frames for context window".format(context_window_size)) 

    while pos < frame_count:
        frame = np.zeros((*frame_size[::-1], 3), np.uint8)

        ctx_start_range = max(0, pos-half)
        ctx_end_range = min(frame_count, ctx_start_range+context_window_size)

        for _, faces in bounding_boxes[ctx_start_range:ctx_end_range]:
            draw_on_faces(frame, faces, box_color)

        out.write(frame)
        pos = pos + 1

    out.release()

def main():
    import sys
    filename = Path(sys.argv[1])
    dest = str(filename.with_stem('{}.overlay'.format(filename.stem)))

    logging.debug('Starting capture device for {}'.format(filename))
    cap = cv2.VideoCapture(str(filename))
    proccesor = VideoProcessor(cap, max=30)
    vdata = inspect_video(cap)
    logging.debug('vdata={}'.format(vdata))

    bounding_boxes = []
    with Stopwatch('VideoProcessor'):
        bounding_boxes = proccesor.work()

    dump_state(filename.with_suffix('.bboxes.bin'), bounding_boxes)

    logging.debug('Write the overlay video')
    with Stopwatch('WriteOverlay'):
        write_overlay(dest, vdata, bounding_boxes, context_window_size=30)

if __name__ == '__main__':
    with Stopwatch('program', level=logging.INFO):
        main()
