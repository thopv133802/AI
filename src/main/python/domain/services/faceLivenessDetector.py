from typing import List

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.saving import model_from_json
from tensorflow.python.keras.utils import np_utils

from src.main.python.core.pathUtils import PathUtils


class FaceLivenessDetector:
    def detectLiveness(self, faces: List):
        pass
    def saveModel(self, folderPath):
        pass
    def loadModel(self, folderPath):
        pass
class FaceLivenessDetectorImpl(FaceLivenessDetector):

    def loadModel(self, folderPath):
        with open(PathUtils.join(folderPath, "model.json"), "r") as file:
            self._model = model_from_json(file.read())
        self._model.load_weights(PathUtils.join(folderPath, "model.h5"))
    def saveModel(self, folderPath):
        with open(PathUtils.join(folderPath, "model.json"), "w") as file:
            file.write(self._model.to_json())
        self._model.save_weights(PathUtils.join(folderPath, "model.h5"))
    def buildModel(self):
        numEpochs = 50

        model = Sequential()
        model.add(Conv2D(16, (3, 3), padding="same", input_shape=(32, 32, 3)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=-1))
        model.add(Conv2D(16, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=-1))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(2))
        model.add(Activation("softmax"))

        model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(lr=1e-4, decay=1e-4 / numEpochs),
            metrics=["accuracy"]
        )

        self._model = model
        self.saveModel(PathUtils.getFaceLivenessFolderPath())

    def trainModel(self):
        numEpochs = 50
        batchSize = 4


        X = []
        y = []

        import os
        for rootPath, _, files in os.walk(PathUtils.join(PathUtils.getFaceLivenessFolderPath(), "real")):
            for fileName in files:
                image = cv2.imread(PathUtils.join(rootPath, fileName))
                image = cv2.resize(image, (32, 32))
                X.append(image)
                y.append(1)

        for rootPath, _, files in os.walk(PathUtils.join(PathUtils.getFaceLivenessFolderPath(), "fake")):
            for fileName in files:
                image = cv2.imread(PathUtils.join(rootPath, fileName))
                image = cv2.resize(image, (32, 32))
                X.append(image)
                y.append(0)

        X = np.array(X)
        y = np.array(y)

        labelEncoder = LabelEncoder()
        y = labelEncoder.fit_transform(y)
        y = np_utils.to_categorical(y, 2)

        trainX, testX, trainY, testY = train_test_split(X, y, test_size = 0.25, random_state = 12345)

        aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                                 width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                                 horizontal_flip=True, fill_mode="nearest")

        self._model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(lr=1e-4, decay=1e-4 / numEpochs),
            metrics=["accuracy"]
        )

        self._model.fit_generator(aug.flow(trainX, trainY, batch_size = batchSize),
                                  validation_data = (testX, testY),
                                  steps_per_epoch = len(trainX) // batchSize,
                                  epochs = numEpochs
        )
        self.saveModel(PathUtils.getFaceLivenessFolderPath())
        
        
    def detectLiveness(self, faces: List):
        return False
        # batchSize = 4
        # faces = np.array([cv2.resize(face, (32, 32)) for face in faces])
        # predictions = self._model.predict(faces, batch_size = batchSize)
        # predictions = [np.argmax(prediction) for prediction in predictions]
        # for prediction in predictions:
        #     if(prediction == 0):
        #         return True
        # return False


# if(__name__ == "__main__"):
#     detector = FaceLivenessDetectorImpl()
#     detector.buildModel()
#     detector.trainModel()
