import logging
import numpy as np
import cv2
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s - %(name)s[%(processName)s|%(filename)s:%(lineno)d] %(message)s')

# patch numpy
setattr(np, 'int', int)

import video_processing
import multiprocessing

from stopwatch import Stopwatch
from utils import dump_state

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
        logging.debug("Writing overlay for frame {}".format(pos)) 
        frame = np.zeros((*frame_size[::-1], 3), np.uint8)

        ctx_start_range = max(0, pos-half)
        ctx_end_range = min(frame_count, ctx_start_range+context_window_size)

        for _, faces in bounding_boxes[ctx_start_range:ctx_end_range]:
            video_processing.draw_on_faces(frame, faces, box_color)

        out.write(frame)
        pos = pos + 1

    out.release()


def process_video(cap, tasks, results):
    num_consumers = multiprocessing.cpu_count()
    logging.debug('Starting multi-processing task with {} consumers'.format(num_consumers))

    detector = video_processing.InsightFaceDetector()
    for w in range(num_consumers):
        video_processing.Consumer(tasks, results, detector=detector).start()

    num_jobs = 0
    da_max = 100
    for frame_num, frame in video_processing.FrameIterator(cap):
        logging.debug('Put frame {} in the queue'.format(frame_num))
        tasks.put((frame_num, frame))
        num_jobs += 1
        if num_jobs > da_max:
            break
            pass # break

    logging.debug('clear the pointer')
    cap.release()

    # Add a poison pill for each consumer
    logging.debug('Send poison pills')
    for i in range(num_consumers):
        tasks.put(None)

    return num_jobs



def main():
    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()

    import sys
    filename = Path(sys.argv[1])

    dest = str(filename.with_stem('{}.overlay'.format(filename.stem)))

    logging.debug('Starting capture device for {}'.format(filename))
    cap = cv2.VideoCapture(str(filename))
    vdata = video_processing.inspect_video(cap)

    # bounding_boxes = load_state(filename.with_suffix('.bboxes.bin'))
    logging.debug('vdata={}'.format(vdata))

    stopwatch = Stopwatch()
    with stopwatch:
        num_jobs = process_video(cap, tasks, results)

        # Wait for all of the tasks to finish
        logging.debug('Wait for the work to end')
        tasks.join()
    logging.info('Finished processing in {}'.format(stopwatch.duration))

    logging.debug('Grabbing the results')
    with stopwatch:
        bounding_boxes = []
        while num_jobs:
            frame_num, faces = results.get()
            bounding_boxes.append((frame_num, faces))
            num_jobs -= 1
    logging.info('Finished Grabbing in {}'.format(stopwatch.duration))

    logging.debug('Sort bouncing boxes by frame')
    # Sort the frames after the fact
    bounding_boxes.sort(key=lambda x: x[0])
    dump_state(filename.with_suffix('.bboxes.bin'), bounding_boxes)

    logging.debug('Write the overlay video')
    with stopwatch:
        write_overlay(dest, vdata, bounding_boxes, context_window_size=30)
    logging.info('Finished overlay in {}'.format(stopwatch.duration))


if __name__ == '__main__':
    with Stopwatch('program'):
        main()
