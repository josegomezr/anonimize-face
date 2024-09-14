import logging

from multiprocessing import Process as WorkerBase
from stopwatch import Stopwatch
from insight_face_detector import InsightFaceDetector

# Use this one for threads instead of processes
# from multiprocessing.dummy import Process as WorkerBase

class FaceRecognitionTask(WorkerBase):
    def __init__(self, task_queue, result_queue, detector=None, logger=None):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.detector = detector

    def process_frame(self, frame_num, frame):
        logging.debug('[f:{}] got a frame'.format(frame_num))
        faces = []

        with Stopwatch('[f:{}] processing frame'.format(frame_num)):
            faces += self.detector.find_faces(frame)

        return frame_num, faces

    def task_generator(self):
        while True:
            logging.debug('Fetching an item from the queue')
            item = self.task_queue.get()
            if item is None:
                logging.debug('EoQ: End of the queue!')
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
            result = self.process_frame(*item)
            self.task_queue.task_done()
            self.result_queue.put(result)