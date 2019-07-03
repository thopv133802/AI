from typing import List
import cv2
import numpy as np
from src.main.python.domain.models.face import Face
from src.main.python.domain.models.image import Image


class FaceDetector:
    def detectFaces(self, image: Image) -> List[Face]:
        pass

class FaceDetectorImpl(FaceDetector):
    def __init__(self, faceDetectorModelPath, faceDetectorConfigPath, confidenceThreshold = 0.8):
        self._confidenceThreshold = confidenceThreshold
        self._frontFaceClassifier = cv2.dnn.readNetFromTensorflow(faceDetectorModelPath, faceDetectorConfigPath)

    def detectFaces(self, image: Image) -> List[Face]:
        (imageHeight, imageWidth) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), [104, 177, 123])
        self._frontFaceClassifier.setInput(blob)
        predictions = self._frontFaceClassifier.forward()

        detectedFaces = []

        for index in range(0, predictions.shape[2]):
            confidence = predictions[0, 0, index, 2]
            if confidence > self._confidenceThreshold:
                box = predictions[0, 0, index, 3:7] * np.array([imageWidth, imageHeight, imageWidth, imageHeight])
                box = box.astype("int")
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                x1 = max(0, min(x1, imageWidth - 1))
                x2 = max(min(x2, imageWidth - 1), 0)
                y1 = max(0, min(y1, imageHeight - 1))
                y2 = max(min(y2, imageHeight - 1), 0)
                detectedFace = Face(x1, y1, x2, y2, image[y1: y2, x1: x2])  # out of box --> throw Exception
                detectedFaces.append(detectedFace)
        return detectedFaces