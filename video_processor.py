import multiprocessing
import logging
from face_recognition_task import FaceRecognitionTask
from insight_face_detector import InsightFaceDetector
from stopwatch import Stopwatch
from utils import FrameIterator, inspect_video


class VideoProcessor:
    def __init__(self, cap, concurrency=None, logger=None, max=None):
        self.concurrency = concurrency or multiprocessing.cpu_count()
        self.frames_queue = multiprocessing.JoinableQueue()
        self.results_queue = multiprocessing.Queue()
        self.cap = cap
        self.max = max
        self.logger = logger or logging.getLogger(__name__)

    def work(self):
        vdata = inspect_video(self.cap)

        self.logger.debug(
            "Starting multi-processing task with {} FaceRecognitionTasks".format(
                self.concurrency
            )
        )

        detector = InsightFaceDetector()
        for w in range(self.concurrency):
            FaceRecognitionTask(
                self.frames_queue, self.results_queue, detector=detector
            ).start()

        for frame_num, frame in FrameIterator(self.cap, until=self.max):
            self.logger.debug("Put frame {} in the queue".format(frame_num))
            self.frames_queue.put((frame_num, frame))

        # Add a poison pill for each consumer
        self.logger.debug("Send poison pills")
        for i in range(self.concurrency):
            self.frames_queue.put(None)

        self.logger.info("Waiting for results...")
        self.frames_queue.join()

        bounding_boxes = []
        while not self.results_queue.empty():
            frame_num, faces = self.results_queue.get()
            bounding_boxes.append((frame_num, faces))

        bounding_boxes.sort(key=lambda x: x[0])
        return bounding_boxes
