import logging

from multiprocessing import Process
from stopwatch import Stopwatch
from insight_face_detector import InsightFaceDetector

# Use this one for threads instead of processes
from multiprocessing.dummy import Process as Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
from dataclasses import dataclass

import numpy as np

class FaceRecognitionTask:
    def __init__(self, task_queue, result_queue, detector=None, logger=None, lock=None):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.detector = detector
        self.lock = lock

    def process_frame(self, frame_num, frame):
        logging.debug("[f:{}] got a frame".format(frame_num))
        faces = []

        with Stopwatch("[f:{}] processing frame".format(frame_num)):
            faces += self.detector.find_faces(frame)

        # return frame_num, faces
        return (frame_num, faces)

    def task_generator(self):
        while True:
            logging.debug("Fetching an item from the queue")
            item = self.task_queue.get()
            if item is None:
                logging.debug("EoQ: End of the queue!")
                # Poison pill means shutdown
                self.task_queue.task_done()
                return
            yield item

    def prepare_detector(self):
        if self.detector is None:
            logging.info("Loading detector from scratch")
            self.detector = InsightFaceDetector()

        with self.lock or contextlib.nullcontext():
            self.detector.warmup()

    def run(self):
        logging.info("Booting worker")
        self.prepare_detector()

        for item in self.task_generator():
            self.process_item(item)

    def process_item(self, item):
        result = self.process_frame(*item)
        self.task_queue.task_done()
        self.result_queue.put(result)


class BatchedProcess(FaceRecognitionTask, Process):
    def process_item(self, batch):
        with ThreadPoolExecutor() as pool:
            futures = [pool.submit(self.process_frame,*item) for item in batch]
            for future in as_completed(futures):
                result = future.result()
                self.result_queue.put(result)
        self.task_queue.task_done()

    # def run(self):
    #     logging.info("Booting worker")
    #     self.prepare_detector()

    #     with ThreadPoolExecutor() as pool:
    #         for batch in self.task_generator():
    #             futures = [pool.submit(self.process_frame, *item) for item in batch]
                
    #             for future in as_completed(futures):
    #                 result = future.result()
    #                 self.result_queue.put(result)
    #             self.task_queue.task_done()



class WithThreads(FaceRecognitionTask, Thread):
    pass

class WithProcesses(FaceRecognitionTask, Process):
    pass
