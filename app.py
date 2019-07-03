import json
import os
import pickle
import random
import sys
import time
import typing
from pprint import pprint

import cv2
import imutils
import math
import numpy
from flask import Flask, request, make_response, abort
from flask.json import jsonify
from sklearn.metrics import f1_score

from src.main.python.core.fileUtils import FileUtils
from src.main.python.core.imageUtils import ImageUtils
from src.main.python.core.injectorUtils import getInjector
from src.main.python.core.jsonUtils import MyJsonEncoder
from src.main.python.core.pathUtils import PathUtils
from src.main.python.domain.models.embeddings import Embeddings
from src.main.python.domain.models.user import User


class App:
    def __init__(self):
        self.injector = getInjector()
        self.imageUploader = self.injector.getImageUploader()

    def runTrain(self):
        users = []
        faceNet = self.injector.getFaceNet()
        faceRecognizer = self.injector.getFaceRecognizer()
        imageAugmenter = self.injector.getImageAugmenter()

        imagesFolderPath = PathUtils.join(PathUtils.getResourcePath(), "images")

        for folderName in os.listdir(imagesFolderPath):
            folderPath = PathUtils.join(imagesFolderPath, folderName)
            print(folderPath)
            for _, _, fileNames in os.walk(folderPath):
                images = [cv2.imread(PathUtils.join(folderPath, fileName)) for fileName in fileNames]
                flippedImages = imageAugmenter.flips(images)
                embeddingsList = faceNet.imagesToEmbeddings(images + flippedImages)
                flippedFileNames = ["flipped-" + fileName for fileName in fileNames]
                embeddingsMap = {embeddingsID: embeddings for embeddingsID, embeddings in
                                 zip(fileNames + flippedFileNames, embeddingsList)}
                user = User(folderName, embeddingsMap)
                users.append(
                    user
                )

        for user in users:
            print(user.toString())

        faceRecognizer.addUsers(users)

    def saveRecognizedImages(self, userId, images):
        currentTime = time.time()
        FileUtils.createDirIfNotExists(PathUtils.join(PathUtils.getResourcePath(), "recognizedImages", str(userId)))
        for image in images:
            tempImageID = userId + str(currentTime)
            tempImageName = tempImageID + ".jpg"
            tempImageFilePath = PathUtils.join(PathUtils.getResourcePath(), "recognizedImages", str(userId),
                                               tempImageName)
            cv2.imwrite(tempImageFilePath, image)

            imageID = self.imageUploader.upload(tempImageID, tempImageFilePath)
            imageName = imageID + "-" + str(currentTime) + ".jpg"
            imageFilePath = PathUtils.join(PathUtils.getResourcePath(), "recognizedImages", str(userId), imageName)
            cv2.imwrite(imageFilePath, image)
            os.remove(tempImageFilePath)

    def saveReconsiderImages(self, userId, images):
        currentTime = time.time()
        FileUtils.createDirIfNotExists(PathUtils.join(PathUtils.getResourcePath(), "reconsiderImages", str(userId)))
        fileNames = []
        for image in images:
            tempImageID = userId + str(currentTime)
            tempImageName = tempImageID + ".jpg"
            tempImageFilePath = PathUtils.join(PathUtils.getResourcePath(), "reconsiderImages", str(userId),
                                               tempImageName)
            cv2.imwrite(tempImageFilePath, image)
            imageID = self.imageUploader.upload(tempImageID, tempImageFilePath)
            imageName = imageID + ".jpg"
            imageFilePath = PathUtils.join(PathUtils.getResourcePath(), "reconsiderImages", str(userId), imageName)
            os.remove(tempImageFilePath)
            cv2.imwrite(imageFilePath, image)

            fileNames.append(imageName)
        return fileNames

    def runRecognitionService(self, host, port):
        flaskApp = Flask(__name__)
        faceRecognizer = self.injector.getFaceRecognizer()
        facenet = self.injector.getFaceNet()
        faceRecognizer.loadFromFile(PathUtils.getFaceClassifierFilePath())
        imageAugmenter = self.injector.getImageAugmenter()
        imageAligner = self.injector.getFaceAligner()
        faceLivenessDetector = self.injector.getFaceLivenessDetector()
        faceLivenessDetector.loadModel(PathUtils.getFaceLivenessFolderPath())
        memberService = self.injector.getMemberService()

        @flaskApp.route("/recognizesByFaces", methods=["POST"])
        def recognizesFaces():
            startTime = time.time()
            print("Start read request body at ", str(startTime))
            images = [cv2.imdecode(numpy.frombuffer(file.read(), numpy.uint8), cv2.IMREAD_UNCHANGED) for file in
                      request.files.values()]

            if faceLivenessDetector.detectLiveness(images) == True:
                return abort(400, {
                    "message": "Phát hiện giả mạo. Vui lòng thử lại."
                })

            images = [ImageUtils.increaseBrightness(image) for image in images]
            images = [imageAligner.alignImage(image) for image in images]

            endTime0 = time.time()
            print("End read request body in", (endTime0 - startTime))
            embeddingsList = facenet.imagesToEmbeddings(images)
            faceRecognition = faceRecognizer.recognizes(embeddingsList)
            userId = faceRecognition.userId
            endTime = time.time()
            print("Execute in", str(endTime - startTime))

            self.saveRecognizedImages(userId, images)

            return make_response(jsonify({
                "userId": userId,
                "probability": str(faceRecognition.probability)
            }), 200)

        @flaskApp.route("/reconsidersByFaces/<userId>", methods=["POST"])
        def reconsidersFaces(userId: str):
            print("On Reconsiders by faces of ", userId)
            userId = userId.capitalize()
            images = [cv2.imdecode(numpy.frombuffer(file.read(), numpy.uint8), cv2.IMREAD_UNCHANGED) for file in
                      request.files.values()]
            images = [ImageUtils.increaseBrightness(image) for image in images]
            images = [imageAligner.alignImage(image) for image in images]
            images += imageAugmenter.flips(images)

            fileNames = self.saveReconsiderImages(userId, images)
            embeddingsList = facenet.imagesToEmbeddings(images)

            embeddingsMap = {fileName: embeddings for (fileName, embeddings) in zip(fileNames, embeddingsList)}

            faceRecognizer.addUsers(
                [User(
                    userId,
                    embeddingsMap
                )]
            )
            return make_response(jsonify({
                "status": "true",
                "message": "success"
            }), 200)

        @flaskApp.route("/members/fetchMembers/<ogn>", methods=["GET"])
        def fetchMembers(ogn: str):
            if ogn == "weSave":
                members = memberService.fetchMembers()
                return make_response(json.dumps({
                    "status": "true",
                    "message": "success",
                    "payload": members
                }, cls=MyJsonEncoder), 200)
            return make_response(jsonify(
                {
                    "status": "false",
                    "message": "Not found organization"
                }
            ), 200)

        @flaskApp.route("/members/fetchMember/<ogn>/<userID>", methods=["GET"])
        def fetchMember(ogn: str, userID: str):
            if ogn == "weSave":
                member = memberService.fetchMember(userID)
                return make_response(json.dumps({
                    "status": "true",
                    "message": "success",
                    "payload": member
                }, cls=MyJsonEncoder), 200)

            return make_response(jsonify({
                "status": "false",
                "message": "Not found organization"
            }), 200)

        @flaskApp.route("/members/fetchTimeKeepings/<ogn>/<userID>", methods=["GET"])
        def fetchTimeKeeptings(ogn: str, userID: str):
            if ogn == "weSave":
                timeKeepings = memberService.fetchTimeKeepings(userID)
                return make_response(json.dumps({
                    "status": "true",
                    "message": "success",
                    "payload": timeKeepings
                }, cls=MyJsonEncoder), 200)
            return make_response(jsonify({
                "status": "false",
                "message": "Not found organization"
            }), 200)

        @flaskApp.route("/members/removeFace/<ogn>/<userID>/<faceID>", methods=["POST", "DELETE"])
        def removeFace(ogn: str, userID: str, faceID: str):
            if ogn == "weSave":
                faceRecognizer.removeUserFace(userID, faceID)
                return make_response(json.dumps({
                    "status": "true",
                    "message": "success"
                }), 200)
            return make_response(jsonify({
                "status": "false",
                "message": "Not found organization"
            }), 200)

        @flaskApp.route("/trainer/retrain", methods=["POST"])
        def retrain():
            faceRecognizer.retrain()
            return make_response(jsonify({
                "status": "true",
                "message": "success"
            }), 2000)

        flaskApp.run(host=host, port=port)


if (__name__ == "__main__"):
    app = App()
    app.runTrain()
    app.runRecognitionService(sys.argv[1], sys.argv[2])
