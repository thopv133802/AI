from pprint import pprint
from typing import List, Dict

import numpy as np

from src.main.python.core.listUtils import ListUtils
from src.main.python.domain.models.faceRecognition import FaceRecognition


class PredictionVoter:
    def vote(self, facePredictions: List[FaceRecognition]) -> FaceRecognition:
        pass


class PredictionVoterImpl(PredictionVoter):
    def vote(self, facePredictions: List[FaceRecognition]) -> FaceRecognition:
        userProbabilities: Dict[str, List[float]] = {}
        for facePrediction in facePredictions:
            userId = facePrediction.userId
            probability = facePrediction.probability
            if(userId not in userProbabilities.keys()):
                userProbabilities[userId] = [probability]
            else:
                userProbabilities[userId].append(probability)
        print("User Probabilities: ", userProbabilities)
        userMeanProbabilities = {userId: (ListUtils.sum(probabilities) / len(probabilities)) for (userId, probabilities) in userProbabilities.items()}
        votedUser = None
        votedMeanProbabilities = -1.
        for (userId, meanProbabilities) in userMeanProbabilities.items():
            if(meanProbabilities > votedMeanProbabilities):
                votedMeanProbabilities = meanProbabilities
                votedUser = userId

        return FaceRecognition(votedUser, votedMeanProbabilities)

    # def oldVote(self, facePredictions: List[FaceRecognition]) -> FaceRecognition:
    #     userProbabilities: Dict[str, float] = {}
    #     for facePrediction in facePredictions:
    #         userId = facePrediction.userId
    #         probability = facePrediction.probability
    #         if(userId not in userProbabilities.keys()):
    #             userProbabilities[userId] = probability
    #         else:
    #             userProbabilities[userId] = userProbabilities[userId] + probability
    #     pprint(userProbabilities)
    #     userIds = []
    #     probabilities = []
    #     for (userId, probability) in userProbabilities.items():
    #         userIds.append(userId)
    #         probabilities.append(probability)
    #     maxProbabilityIndex: int = ListUtils.argmax(probabilities)
    #     return FaceRecognition(userIds[maxProbabilityIndex], probabilities[maxProbabilityIndex])



