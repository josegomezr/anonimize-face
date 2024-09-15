import multiprocessing
import threading
import logging
import queue

from insight_face_detector import InsightFaceDetector
from stopwatch import Stopwatch
from utils import FrameIterator, inspect_video
from itertools import islice


from concurrent.futures import ThreadPoolExecutor, as_completed


class VideoProcessor:
    def __init__(self, cap, concurrency=None, logger=None, max=None):
        self.concurrency = concurrency or multiprocessing.cpu_count()
        self.frames_queue = multiprocessing.JoinableQueue()
        self.results_queue = multiprocessing.Queue()
        self.cap = cap
        self.max = max
        self.logger = logger or logging.getLogger(__name__)

        self.pool = None

    def post_work(self):
        bounding_boxes = []
        while not self.results_queue.empty():
            frame_num, faces = self.results_queue.get()
            bounding_boxes.append((frame_num, faces))

        bounding_boxes.sort(key=lambda x: x[0])
        return bounding_boxes

    def tpe_work(self):
        detector = InsightFaceDetector()
        detector.warmup()

        self.pool = ThreadPoolExecutor()
        try:
            logging.info("Scheduling work")
            futures = [
                self.pool.submit(self.process_frame, detector, *item)
                for item in FrameIterator(self.cap, until=self.max)
            ]
            logging.info("Processing...")
            for future in as_completed(futures):
                frame_num, faces = future.result()
                self.results_queue.put((frame_num, faces))
        except KeyboardInterrupt as e:
            logging.info("Shutting down pool")
        finally:
            self.pool.shutdown(cancel_futures=True)

    def process_frame(self, detector, frame_num, frame):
        with Stopwatch("[f:{}] processing frame".format(frame_num)):
            return frame_num, detector.find_faces(frame)

    def graceful_shutdown(self):
        try:
            self.logger.info("Handling graceful shutdown")

            if self.pool:
                self.pool.shutdown(cancel_futures=True)
            self.frames_queue.cancel_join_thread()
            self.frames_queue.close()
        except KeyboardInterrupt as e:
            self.logger.info("!!! Forcefully shutting down !!!")
            raise e

    def work(self):
        try:
            self.logger.debug(
                "Starting FaceRecognitionTask with concurrency={}".format(
                    self.concurrency
                )
            )

            self.tpe_work()

            # Add a poison pill for each consumer
            self.logger.debug("Send poison pills")
            
            self.frames_queue.join()

            return self.post_work()
        except KeyboardInterrupt as e:
            self.graceful_shutdown()
            return []
