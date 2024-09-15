import multiprocessing
import threading
import logging
import queue
from face_recognition_task import WithThreads, WithProcesses, BatchedProcess

from insight_face_detector import InsightFaceDetector
from stopwatch import Stopwatch
from utils import FrameIterator, inspect_video
from itertools import islice


def batched(iterable, n):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


class VideoProcessor:
    def __init__(self, cap, concurrency=None, logger=None, max=None):
        self.concurrency = concurrency or multiprocessing.cpu_count()
        self.frames_queue = multiprocessing.JoinableQueue()
        self.results_queue = multiprocessing.Queue()
        self.cap = cap
        self.max = max
        self.logger = logger or logging.getLogger(__name__)

        self.workers = []

    def post_work(self):
        bounding_boxes = []
        while not self.results_queue.empty():
            frame_num, faces = self.results_queue.get()
            bounding_boxes.append((frame_num, faces))

        bounding_boxes.sort(key=lambda x: x[0])
        return bounding_boxes

    # def batched_process_work(self):
    #     lock = multiprocessing.Lock()
    #     detector = None
    #     for w in range(self.concurrency):
    #         worker = BatchedProcess(
    #             self.frames_queue, self.results_queue, detector=detector, lock=lock
    #         )
    #         self.workers.append(worker)
    #         worker.start()

    #     for batch in batched(FrameIterator(self.cap, until=self.max), 30):
    #         self.frames_queue.put(batch)

    # def serialized_process_work(self):
    #     lock = multiprocessing.Lock()
    #     detector = None
    #     for w in range(self.concurrency):
    #         worker = WithProcesses(
    #             self.frames_queue, self.results_queue, detector=detector, lock=lock
    #         )
    #         self.workers.append(worker)
    #         worker.start()

    #     for frame_num, frame in FrameIterator(self.cap, until=self.max):
    #         self.logger.debug("Put frame {} in the queue".format(frame_num))
    #         self.frames_queue.put((frame_num, frame))

    def threaded_work(self):
        lock = threading.Lock()
        detector = InsightFaceDetector()
        detector.warmup()

        self.concurrency = self.concurrency * 5
        for w in range(self.concurrency):
            worker = WithThreads(
                self.frames_queue, self.results_queue, detector=detector, lock=lock
            )
            self.workers.append(worker)
            worker.start()

        for frame_num, frame in FrameIterator(self.cap, until=self.max):
            self.logger.debug("Put frame {} in the queue".format(frame_num))
            self.frames_queue.put((frame_num, frame))

    def graceful_shutdown(self):
        try:
            self.frames_queue.cancel_join_thread()

            self.logger.info("Handling graceful shutdown")
            self.logger.info("Waiting for workers to stop")
            self.frames_queue.close()

            for i, worker in enumerate(self.workers):
                worker.graceful_stop = True
                worker.join()
                self.logger.debug(
                    "Worker {}/{} stopped".format(i + 1, len(self.workers))
                )
        except KeyboardInterrupt as e:
            self.logger.info("!!! Forcefully shutting down !!!")
            raise e

    def work(self, strategy="threaded"):
        self.logger.debug(
            "Starting FaceRecognitionTask with concurrency={}".format(self.concurrency)
        )

        try:
            self.threaded_work()

            # Add a poison pill for each consumer
            self.logger.debug("Send poison pills")
            for i in range(self.concurrency):
                self.frames_queue.put(None)
                self.logger.info("Waiting for results...")
                self.frames_queue.join()

                return self.post_work()
        except KeyboardInterrupt as e:
            self.graceful_shutdown()
            # import ipdb; ipdb.set_trace()
            return []
