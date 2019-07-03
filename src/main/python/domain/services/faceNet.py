from typing import List
import tensorflow as tf
import math
import numpy as np
import cv2
from tensorflow import GraphDef, Session

from src.main.python.core.imageUtils import ImageUtils
from src.main.python.core.listUtils import ListUtils
from src.main.python.domain.models.embeddings import Embeddings
from src.main.python.domain.models.image import Image


class FaceNet:
    def preprocessImage(self, image: Image) -> Image:
        pass
    def imagesToEmbeddings(self, images: List[Image]) -> List[Embeddings]:
        pass
    def imageToEmbeddings(self, image: Image) -> Embeddings:
        pass

class FaceNetImpl(FaceNet):
    def __init__(self, modelFilePath, batchSize = 4):
        self._batchSize = batchSize
        with tf.gfile.GFile(modelFilePath, "rb") as file:
            model = file.read()
        _graphDef = GraphDef()
        _graphDef.ParseFromString(model)
        tf.import_graph_def(_graphDef)
        self._inputImageSize = 160
        self._inputNode = tf.get_default_graph().get_tensor_by_name("import/input:0")
        self._embeddingsNode = tf.get_default_graph().get_tensor_by_name("import/embeddings:0")
        self._isTrainingNode = tf.get_default_graph().get_tensor_by_name("import/phase_train:0")
        self._embeddingsSize = self._embeddingsNode.get_shape()[1]
        self._session = Session()

    def preprocessImage(self, image: Image) -> Image:
        image =cv2.resize(image, (self._inputImageSize, self._inputImageSize))
        mean = np.mean(image)
        standardDeviation = np.std(image)
        #standardDeviationAdj = np.maximum(standardDeviation, 1.0 / np.sqrt(image.size))
        return np.multiply(np.subtract(image, mean), 1 / standardDeviation)

    def imagesToEmbeddings(self, images: List[Image]) -> List[Embeddings]:
        images = [self.preprocessImage(image) for image in images]

        imagesEmbeddings: List[Embeddings] = ListUtils.createListWithSize(len(images))
        numIterations: int = math.ceil(len(images) / self._batchSize)
        numImages = len(images)

        for iteration in range(numIterations):
            startIndex = iteration * self._batchSize
            endIndex = min(startIndex + self._batchSize, numImages)
            batchImages = images[startIndex: endIndex]
            imagesEmbeddings[startIndex: endIndex] = self._session.run(
                self._embeddingsNode, {
                    self._isTrainingNode: False,
                    self._inputNode: batchImages
                }
            )

        return imagesEmbeddings

    def imageToEmbeddings(self, image: Image) -> Embeddings:
        image = self.preprocessImage(image)
        return self._session.run(self._embeddingsNode, {
            self._isTrainingNode: False,
            self._inputNode: np.array([image])
        })[0]