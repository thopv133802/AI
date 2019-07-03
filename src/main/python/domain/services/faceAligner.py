from typing import Optional
import dlib
from src.main.python.domain.models.eye import Eyes, Eye
from src.main.python.domain.models.face import Face
import cv2

from src.main.python.domain.models.image import Image


class FaceAligner:
    def align(self, face: Face, image: Image) -> Face:
        pass

    def alignImage(self, image: Image) -> Image:
        pass


class EyesDetector:
    def detect(self, face: Face, image: Image) -> Optional[Eyes]:
        pass

    def detectImage(self, image) -> Optional[Eyes]:
        pass


class EyesDetectorImpl(EyesDetector):
    def __init__(self, shapePredictorFilePath):
        self.shapePredict = dlib.shape_predictor(shapePredictorFilePath)

    def detect(self, face: Face, image: Image) -> Optional[Eyes]:
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        predictedShape = self.shapePredict(grayImage, face.toDlibRectangle())

        if predictedShape is not None:
            point1, point2, point3, point4 = predictedShape.part(0), predictedShape.part(1),  predictedShape.part(2),  predictedShape.part(3)
            eye1 = Eye(point1.x, point1.y, point2.x, point2.y)
            eye2 = Eye(point3.x, point3.y, point4.x, point4.y)
            return Eyes(eye1, eye2)
        return None

    def detectImage(self, image) -> Optional[Eyes]:
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (width, height) = image.shape[1], image.shape[0]
        predictedShape = self.shapePredict(grayImage, dlib.rectangle(
            left=0,
            right=width,
            top=0,
            bottom=height,
        ))

        if predictedShape is not None:
            point1, point2, point3, point4 = predictedShape.part(0), predictedShape.part(1), predictedShape.part(
                2), predictedShape.part(3)
            eye1 = Eye(point1.x, point1.y, point2.x, point2.y)
            eye2 = Eye(point3.x, point3.y, point4.x, point4.y)
            return Eyes(eye1, eye2)
        return None


class FaceAlignerImpl(FaceAligner):
    def __init__(self, eyesDetector: EyesDetector):
        self.eyesDetector = eyesDetector

    def align(self, face: Face, image: Image) -> Face:
        eyes = self.eyesDetector.detect(face, image)

        if eyes is not None:
            angle = eyes.calculateAngle()
            M = cv2.getRotationMatrix2D((face.xCenter, face.yCenter), angle, scale=1)
            (imageHeight, imageWidth) = (image.shape[0], image.shape[1])
            alignedImage = cv2.warpAffine(image, M, (imageWidth, imageHeight), flags=cv2.INTER_CUBIC)
            return Face(
                face.x1,
                face.y1,
                face.x2,
                face.y2,
                alignedImage[face.y1: face.y2, face.x1: face.x2]
            )
        return face

    def alignImage(self, image: Image) -> Image:
        eyes = self.eyesDetector.detectImage(image)
        if eyes is not None:
            angle = eyes.calculateAngle()
            (xCenter, yCenter) = eyes.calculateCenter()
            M = cv2.getRotationMatrix2D((xCenter, yCenter), angle, scale=1)
            (imageHeight, imageWidth) = (image.shape[0], image.shape[1])
            alignedImage = cv2.warpAffine(image, M, (imageWidth, imageHeight), flags=cv2.INTER_CUBIC)
            return alignedImage

        return image
