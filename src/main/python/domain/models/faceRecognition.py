import math

class FaceRecognition:
    def __init__(self, userId: str, probability: float):
        self.userId = userId
        self.probability = probability
    def isConfidence(self) -> bool:
        return self.probability > 0.45
    def toString(self):
        print("Probability", self.probability)
        return self.userId + " - " + str(self.probability)[0:6]