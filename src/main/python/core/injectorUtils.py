from src.main.python.core.pathUtils import PathUtils
from src.main.python.domain.services.imageUploader import ImageUploader, ImageUploaderImpl
from src.main.python.domain.services.memberService import MemberService, MemberServiceImpl
from src.main.python.domain.services.faceAligner import FaceAligner, FaceAlignerImpl, EyesDetector, EyesDetectorImpl
from src.main.python.domain.services.faceDetector import FaceDetector, FaceDetectorImpl
from src.main.python.domain.services.faceLivenessDetector import FaceLivenessDetector, FaceLivenessDetectorImpl
from src.main.python.domain.services.faceNet import FaceNet, FaceNetImpl
from src.main.python.domain.services.faceRecognizer import FaceRecognizer, FaceRecognizerImpl
from src.main.python.domain.services.imageAugmenter import ImageAugmenter, ImageAugmenterImpl
from src.main.python.domain.services.imageOptimizer import ImageOptimizer, ImageOptimizerImpl
from src.main.python.domain.services.predictionVoter import PredictionVoter, PredictionVoterImpl


class Injector:
    def __init__(self):
        self._faceNet = None
        self._faceRecognizer = None
        self._faceAligner = None
        self._faceDetector = None
        self._resourcePaths = None
        self._eyesDetector = None
        self._imageAugmenter = None
        self._predictionVoter = None
        self._imageOptimizer = None
        self._faceLivenessDetector = None
        self._memberService = None
        self._imageUploader = None
        pass

    def getImageUploader(self) -> ImageUploader:
        if self._imageUploader is None:
            self._imageUploader = ImageUploaderImpl()
        return self._imageUploader
    def getFaceLivenessDetector(self) -> FaceLivenessDetector:
        if(self._faceLivenessDetector is None):
            self._faceLivenessDetector = FaceLivenessDetectorImpl()
        return self._faceLivenessDetector
    def getImageAugmenter(self) -> ImageAugmenter:
        if (self._imageAugmenter is None):
            self._imageAugmenter = ImageAugmenterImpl()
        return self._imageAugmenter

    def getFaceRecognizer(self) -> FaceRecognizer:
        if (self._faceRecognizer is None):
            self._faceRecognizer = FaceRecognizerImpl(predictionVoter=self.getPredictionVoter())
        return self._faceRecognizer

    def getImageOptimizer(self) -> ImageOptimizer:
        if (self._imageOptimizer is None):
            self._imageOptimizer = ImageOptimizerImpl()
        return self._imageOptimizer

    def getPredictionVoter(self) -> PredictionVoter:
        if (self._predictionVoter is None):
            self._predictionVoter = PredictionVoterImpl()
        return self._predictionVoter

    def getResourcePaths(self) -> PathUtils:
        if (self._resourcePaths is None):
            self._resourcePaths = PathUtils()
        return self._resourcePaths

    def getFaceDetector(self) -> FaceDetector:
        if (self._faceDetector is None):
            self._faceDetector = FaceDetectorImpl(
                faceDetectorModelPath=PathUtils.getFaceDetectorModelPath(),
                faceDetectorConfigPath=PathUtils.getFaceDetectorConfigPath()
            )
        return self._faceDetector

    def getFaceAligner(self) -> FaceAligner:
        if (self._faceAligner is None):
            self._faceAligner = FaceAlignerImpl(
                eyesDetector=self.getEyesDetector()
            )
        return self._faceAligner

    def getEyesDetector(self) -> EyesDetector:
        if (self._eyesDetector is None):
            self._eyesDetector = EyesDetectorImpl(
                shapePredictorFilePath=PathUtils.getShapePredictorFilePath()
            )
        return self._eyesDetector

    def getFaceNet(self) -> FaceNet:
        if (self._faceNet is None):
            self._faceNet = FaceNetImpl(
                modelFilePath=PathUtils.getFaceNetModelFilePath()
            )
        return self._faceNet

    def getMemberService(self) -> MemberService:
        if(self._memberService is None):
            self._memberService = MemberServiceImpl()
        return self._memberService


injector = Injector()


def getInjector() -> Injector:
    return injector
