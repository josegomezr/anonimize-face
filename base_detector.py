class BaseDetector:
    def __init__(
        self,
        threshold=0.3,
    ):
        self.threshold = threshold

    def find_faces(self, frame):
        raise NotImplementedError("")

    def warmup(self):
        raise NotImplementedError("")
