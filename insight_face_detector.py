import numpy as np

setattr(np, "int", int)

from insightface.app import FaceAnalysis


class InsightFaceDetector:
    def __init__(
        self,
        threshold=0.3,
    ):
        self.app = FaceAnalysis(
            providers=["CPUExecutionProvider"],
            # providers=["CUDAExecutionProvider"],
            # providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            allowed_modules=["detection"],
            det_thresh=0.3,
        )
        self.app.prepare(ctx_id=0)
        self.cold = True

    def find_faces(self, frame):
        faces = self.app.get(frame)
        all_faces = []
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(int).tolist()
            all_faces.append(box)
        return all_faces

    def warmup(self):
        if not self.cold:
            return
        self.cold = True
        frame = np.random.randint(0, 255, (480, 640, 3)).astype("uint8")
        self.find_faces(frame)
