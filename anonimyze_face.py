#!/bin/env python

import argparse
import os

# I hate auto updates... stop doing things I'm not telling you to.
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import logging
import numpy as np
import cv2
from pathlib import Path

from video_processor import VideoProcessor

from stopwatch import Stopwatch
from utils import dump_state, load_state, inspect_video, draw_on_faces

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(asctime)s - %(name)s[%(processName)s#%(thread)x|%(filename)s:%(lineno)d] %(message)s",
)


def write_overlay_standalone(dest, vdata, bounding_boxes, context_window_size=15):
    frame_size = vdata["frame_size"]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(dest), fourcc, vdata["fps"], frame_size)
    box_color = (0, 255, 0)

    pos = 0
    frame_count = len(bounding_boxes)

    half = context_window_size // 2

    logging.debug("Using {} frames for context window".format(context_window_size))

    while pos < frame_count:
        frame = np.zeros((*frame_size[::-1], 3), np.uint8)

        for _, faces in sliding_overlay(
            bounding_boxes, pos, context_window_size=context_window_size
        ):
            draw_on_faces(frame, faces, box_color)

        out.write(frame)
        pos = pos + 1

    out.release()


def sliding_overlay(bounding_boxes, start, context_window_size=15):
    frame_count = len(bounding_boxes)

    half = context_window_size // 2
    ctx_start_range = max(0, start - half)
    ctx_end_range = min(frame_count, ctx_start_range + context_window_size)

    ctx_start_range = int(ctx_start_range)
    ctx_end_range = int(ctx_end_range)

    return iter(bounding_boxes[ctx_start_range:ctx_end_range])


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "files",
        metavar="video file",
        type=str,
        nargs="+",
        help="list of video files to be processed",
    )
    parser.add_argument(
        "--data-only",
        action=argparse.BooleanOptionalAction,
        help="only dump bounding boxes",
        default=False,
    )
    parser.add_argument(
        "--write-overlay",
        action=argparse.BooleanOptionalAction,
        help="write overlay video (black background, box_color [green hardocded] foreground)",
        default=True,
    )
    parser.add_argument(
        "--write-merged",
        action=argparse.BooleanOptionalAction,
        help="write overlay and original in a separate video (no audio supported, use ffmpeg_merger.bash for audio support)",
        default=False,
    )
    parser.add_argument(
        "--use-existing-bbox",
        action=argparse.BooleanOptionalAction,
        help="skip face detection if bbox exists",
        default=False,
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    for filepath in args.files:
        filename = Path(filepath)

        logging.info("Processing {}".format(filename))

        bbox_filename = filename.with_suffix(".bboxes.bin")
        cap = cv2.VideoCapture(str(filename))

        vdata = inspect_video(cap)
        logging.info("vdata={}".format(vdata))
        processor = VideoProcessor(cap)

        if bbox_filename.exists() and args.use_existing_bbox:
            logging.info("Using previous frames from {}".format(bbox_filename))
            bounding_boxes = load_state(bbox_filename)
        else:
            logging.debug("Starting capture device for {}".format(filename))

            bounding_boxes = []
            with Stopwatch("VideoProcessor"):
                bounding_boxes = processor.work()

            logging.info("Dumping bounding boxes to {}".format(bbox_filename))
            dump_state(bbox_filename, bounding_boxes)

            if args.data_only:
                break

        if args.write_overlay:
            dest = str(filename.with_stem("{}.overlay".format(filename.stem)))
            logging.info("Writing overlay video in {}".format(dest))
            with Stopwatch("WriteOverlay"):
                write_overlay_standalone(
                    dest, vdata, bounding_boxes, context_window_size=30
                )

        if args.write_merged:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            dest = str(filename.with_stem("{}.merged".format(filename.stem)))
            logging.info("Writing merged video in {}".format(dest))

            frame_size = vdata["frame_size"]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(dest), fourcc, vdata["fps"], frame_size)
            box_color = (0, 255, 0)

            for frame_num, frame in processor.frames():
                for _, faces in sliding_overlay(
                    bounding_boxes, frame_num, context_window_size=30
                ):
                    draw_on_faces(frame, faces, box_color)
                out.write(frame)

            out.release()

        cap.release()


if __name__ == "__main__":
    with Stopwatch("program", level=logging.DEBUG):
        main()
