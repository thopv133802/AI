import os
from pprint import pprint
from typing import List, Dict

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

from src.main.python.core.fileUtils import FileUtils
from src.main.python.core.listUtils import ListUtils
from src.main.python.core.pathUtils import PathUtils
from src.main.python.domain.models.embeddings import Embeddings
from src.main.python.domain.models.faceRecognition import FaceRecognition
from src.main.python.domain.models.user import User
from src.main.python.domain.services.predictionVoter import PredictionVoter


class FaceRecognizer:
    def retrain(self):
        pass

    def recognizes(self, embeddingsList: List[Embeddings]) -> FaceRecognition:
        pass

    def addUsers(self, users: List[User]):
        pass

    def removeUserFace(self, userID, faceID):
        pass

    def saveToFile(self, filePath: str):
        pass

    def loadFromFile(self, filePath: str):
        pass


class FaceRecognizerImpl(FaceRecognizer):
    def removeUserFace(self, userID, faceID):
        faceImageFilePath = PathUtils.join(PathUtils.getResourcePath(), "reconsiderImages", userID, faceID)
        os.remove(faceImageFilePath)
        self._users[userID].removeEmbeddings(faceID)
        self._train()

    def __init__(self, predictionVoter: PredictionVoter):
        self._predictionVoter = predictionVoter
        self._classifier = None
        self._users: Dict[str, User] = {}

    def retrain(self):
        self._train()

    def _train(self):
        newClassifier = SVC(kernel="linear", probability=True, random_state=12345)
        # newClassifier = AdaBoostClassifier(
        #     SVC(kernel = "linear", probability = True),
        #     n_estimators = 1
        # )
        if len(self._users) < 2:
            print("Length users lesser than 2...")
            print("--> Don't train anything...")
        else:
            print("Start train.")
            users = self._users.values()

            inputs: List[Embeddings] = []
            labels: List[str] = []

            for user in users:
                inputs += user.getEmbeddingsList()
                labels += [user.userId] * len(user.getEmbeddingsList())

            newClassifier.fit(inputs, labels)
        self._classifier = newClassifier
        print("FaceRecognizer: Trained.")
        self.saveToFile(PathUtils.getFaceClassifierFilePath())

    def recognizes(self, embeddingsList: List[Embeddings]) -> FaceRecognition:
        lenUsers = len(self._users.keys())
        if lenUsers == 0:
            return FaceRecognition(
                "Không nhận ra",
                0.0
            )
        elif (lenUsers == 1):
            user = list(self._users.values())[0]
            return FaceRecognition(
                user.userId,
                1.0
            )

        print("FaceRecognizer: Start recognizes")
        predictions: List[float] = self._classifier.predict_proba(embeddingsList)

        pprint(predictions)

        labels: List[str] = self._classifier.classes_

        indices = [ListUtils.argmax(prediction) for prediction in predictions]
        predictedLabels = [labels[index] for index in indices]

        probabilities = [ListUtils.max(prediction) for prediction in predictions]

        facePredictions = [FaceRecognition(predictedLabel, probability) for (predictedLabel, probability) in
                           zip(predictedLabels, probabilities)]

        votedFacePrediction = self._predictionVoter.vote(facePredictions)

        return votedFacePrediction

    def addUsers(self, users: List[User]):
        print("FaceRecognizer: Start add users")
        for user in users:
            if user.userId in self._users.keys() is not None:
                user.merge(self._users[user.userId])
            self._users[user.userId] = user
        self._train()

        print("FaceRecognizer: Users added")

    def saveToFile(self, filePath: str):
        print("FaceRecognizer: Start save to file")
        FileUtils.saveObject((self._classifier, self._users), filePath)
        print("FaceRecognizer: Saved to file")

    def loadFromFile(self, filePath: str):
        print("FaceRecognizer: Start load from file")
        self._classifier, self._users = FileUtils.loadObject(filePath)
        for user in self._users.values():
            print(user.toString())
        print("FaceRecognizer: Loaded from file")
