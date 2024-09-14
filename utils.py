import cv2
import pickle
import json


def dump_state(name, obj, marshaller="pickle"):
    mode = "wb+"
    dumper = pickle

    if marshaller == "json":
        mode = "w+"
        dumper = json

    with open(name, mode) as f:
        dumper.dump(obj, f)


def load_state(name, marshaller="pickle"):
    mode = "rb+"
    dumper = pickle

    if marshaller == "json":
        mode = "r+"
        dumper = json

    with open(name, mode) as f:
        return dumper.load(f)


def concat_images(img, *rest):
    height, width = img.shape[:2]
    method = cv2.hconcat
    if width > height:
        method = cv2.vconcat

    return method([img, *rest])


def draw_on_faces(img, faces, box_color, shape="rect"):
    for bbox in faces:
        if shape == "circle":
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            radius = (max(abs(bbox[0] - bbox[2]), abs(bbox[1] - bbox[3]))) // 2
            cv2.circle(img, (cx, cy), radius, box_color, -1)
        else:
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, -1)


def inspect_video(cap):
    amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Calculate the duration of the video in seconds
    duration = amount_of_frames / fps
    codec = cap.get(cv2.CAP_PROP_FOURCC)

    return {
        "frame_count": amount_of_frames,
        "frame_size": frame_size,
        "fps": fps,
        "duration": duration,
        "codec": int(codec),
    }


class FrameIterator:
    def __init__(self, cap, until=None):
        self.cap = cap
        self.until = until

    def __iter__(self):
        self.frame_num = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        return self

    def __next__(self):
        if self.until and self.frame_num >= self.until:
            raise StopIteration

        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration

        current_frame = self.frame_num
        self.frame_num += 1
        return current_frame, frame
