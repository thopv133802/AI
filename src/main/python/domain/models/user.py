from typing import List, Dict

from src.main.python.domain.models.embeddings import Embeddings
from src.main.python.domain.models.image import Image


class User:
    def __init__(self, userId: str, embeddingsMap: Dict[str, Embeddings]):
        self.userId = userId
        self.embeddingsMap = embeddingsMap

    def getEmbeddingsList(self):
        return self.embeddingsMap.values()

    def merge(self, user):
        self.embeddingsMap.update(user.embeddingsMap)

    def toString(self) -> str:
        return "User ID: " + self.userId + ", Embeddings list size: " + str(len(self.embeddingsMap.keys()))

    def removeEmbeddings(self, embeddingsID):
        del self.embeddingsMap[embeddingsID]
