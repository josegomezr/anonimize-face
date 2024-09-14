import cv2
import insightface
from insightface.app import FaceAnalysis
import multiprocessing
import logging
from multiprocessing import Process as RealProcess
from multiprocessing.dummy import Process as ThreadProcess, Pool as ThreadPool
from stopwatch import Stopwatch

WorkerBase = RealProcess
# comment this for multiprocessing
# WorkerBase = ThreadProcess

class InsightFaceDetector:
    def __init__(self):
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], allowed_modules=['detection'], det_thresh=0.3)
        self.app.prepare(ctx_id=0)

    def find_faces(self, frame):
        faces = self.app.get(frame)
        all_faces = []
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(int).tolist()
            all_faces.append(box)
        return all_faces

def inspect_video(cap):
    amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Calculate the duration of the video in seconds
    duration = amount_of_frames / fps
    codec = cap.get(cv2.CAP_PROP_FOURCC)

    return {
        'frame_count': amount_of_frames,
        'frame_size': frame_size,
        'fps': fps,
        'duration': duration,
        'codec': int(codec)
    }

def concat_images(img, *rest):
    height, width = img.shape[:2]
    method = cv2.hconcat
    if width > height:
        method = cv2.vconcat

    return method([img, *rest])


def draw_on_faces(img, faces, box_color, shape='rect'):
    for bbox in faces:
        if shape == 'circle':
            cx = (bbox[0]+bbox[2]) // 2
            cy = (bbox[1]+bbox[3]) // 2
            radius = (max(abs(bbox[0]-bbox[2]), abs(bbox[1]-bbox[3]))) // 2
            cv2.circle(img, (cx, cy), radius, box_color, -1)
        else:
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, -1)


class FrameIterator:
    def __init__(self, cap):
        self.cap = cap
    def __iter__(self):
        self.frame_num = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        return self
    def __next__(self):
        # if not self.cap.isOpened():
        #     raise StopIteration
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration
        current_frame = self.frame_num
        self.frame_num += 1
        return current_frame, frame



class Consumer(WorkerBase):
    def __init__(self, task_queue, result_queue, detector=None):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.detector = detector

    def process_frame(self, frame_num, frame):
        faces = []

        with Stopwatch('[f:{}] processing frame'.format(frame_num)):
            faces += self.detector.find_faces(frame)

        return faces

    def task_generator(self):
        while True:
            logging.debug('Fetching an item from the queue')
            item = self.task_queue.get()
            if item is None:
                logging.info('EoQ: End of the queue!')
                # Poison pill means shutdown
                self.task_queue.task_done()
                return
            yield item

    def run(self):
        logging.info('Booting worker')
        if self.detector is None:
            logging.info('Loading detector from scratch')
            self.detector = InsightFaceDetector()

        for item in self.task_generator():
            frame_num, frame = item
            logging.debug('[f:{}] got a frame'.format(frame_num))
            faces = self.process_frame(frame_num, frame)
            self.task_queue.task_done()
            self.result_queue.put((frame_num, faces))

