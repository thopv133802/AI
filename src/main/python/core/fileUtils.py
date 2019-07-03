import os
import pickle


class FileUtils:
    @staticmethod
    def saveObject(object, filePath):
        with open(filePath, "wb") as file:
            pickle.dump(object, file)
    @staticmethod
    def loadObject(filePath):
        with open(filePath, "rb") as file:
            return pickle.load(file)

    @staticmethod
    def exists(filePath):
        return os.path.exists(filePath)

    @staticmethod
    def createDirIfNotExists(path):
        if(not FileUtils.exists(path)):
            os.makedirs(path)