import os
import cv2 as cv
import numpy as np
from base_detector import BaseDetector
from yunet import YuNet


class YunetDetector(BaseDetector):
    CPU_BACKEND = cv.dnn.DNN_BACKEND_OPENCV
    CUDA_BACKEND = cv.dnn.DNN_BACKEND_CUDA
    CUDA_FP16_BACKEND = cv.dnn.DNN_BACKEND_CUDA
    BACKEND_TIMVX = cv.dnn.DNN_BACKEND_TIMVX
    BACKEND_CANN = cv.dnn.DNN_BACKEND_CANN

    BACKEND_TARGET_PAIRS = {
        CPU_BACKEND: cv.dnn.DNN_TARGET_CPU,
        CUDA_BACKEND: cv.dnn.DNN_TARGET_CUDA,
        CUDA_FP16_BACKEND: cv.dnn.DNN_TARGET_CUDA_FP16,
        BACKEND_TIMVX: cv.dnn.DNN_TARGET_NPU,
        BACKEND_CANN: cv.dnn.DNN_TARGET_NPU,
    }

    def __init__(self, threshold=0.4, backend_id=CUDA_BACKEND):
        super().__init__(threshold=threshold)
        target_id = self.BACKEND_TARGET_PAIRS[backend_id]
        # model available at OpenCV Zoo:
        # https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
        # TODO: make it download like insight face

        if not os.path.exists("face_detection_yunet_2023mar.onnx"):
            raise RuntimeError(
                "Model not found on root, download it from opencv_zoo.\nhttps://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            )

        self.model = YuNet(
            modelPath="face_detection_yunet_2023mar.onnx",
            inputSize=[640, 480],
            confThreshold=threshold,
            nmsThreshold=threshold,
            topK=5000,
            backendId=backend_id,
            targetId=target_id,
        )
        self.cold = True

    def find_faces(self, frame):
        self.model.setInputSize(frame.shape[:2][::-1])
        faces = self.model.infer(frame)

        all_faces = []
        for det in faces:
            if any(det < 0):
                continue
            x, y, w, h = det.astype(int).tolist()[:4]
            box = (x, y, (x + w), (y + h))
            all_faces.append(box)
        return all_faces

    def warmup(self):
        if not self.cold:
            return
        self.cold = True
        frame = np.random.randint(0, 255, (480, 640, 3)).astype("uint8")
        self.find_faces(frame)
