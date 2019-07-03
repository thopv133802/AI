import os
from typing import List

from src.main.python.core.fileUtils import FileUtils
from src.main.python.core.pathUtils import PathUtils


class TimeKeeping:
    def __init__(self, id_: str, face: str, created: int):
        self.id = id_
        self.face = face
        self.created = created


class RemoteImage:
    def __init__(self, _id: str, src: str):
        self.id = _id
        self.src = src


class Member:
    def __init__(self, userId: str, faces: List[RemoteImage]):
        self.userId = userId
        self.faces = faces


class MemberService:
    def fetchMembers(self) -> List[Member]:
        pass

    def fetchMember(self, userID):
        pass

    def fetchTimeKeepings(self, userID) -> List[TimeKeeping]:
        pass



class MemberServiceImpl(MemberService):

    def fetchMembers(self) -> List[Member]:
        reconsiderImagesFolderPath = PathUtils.join(PathUtils.getResourcePath(), "reconsiderImages")
        members = []
        for folderName in os.listdir(reconsiderImagesFolderPath):
            userId = folderName
            folderPath = PathUtils.join(reconsiderImagesFolderPath, folderName)
            for rootPath, _, files in os.walk(folderPath):
                files = files[:3]
                images = [RemoteImage(fileName, "https://i.imgur.com/" + fileName) for fileName in files]
                members.append(Member(userId, images))
                break
        return members

    def fetchMember(self, userID):
        memberFolderPath = PathUtils.join(PathUtils.getResourcePath(), "reconsiderImages", userID)
        for _, _, files in os.walk(memberFolderPath):
            images = [RemoteImage(fileName, "https://i.imgur.com/" + fileName) for fileName in files]
            return Member(userID, images)

    def fetchTimeKeepings(self, userID) -> List[TimeKeeping]:
        memberFolderPath = PathUtils.join(PathUtils.getResourcePath(), "recognizedImages", userID)
        timeKeepings = []
        for _, _, files in os.walk(memberFolderPath):
            for fileName in files:
                fileName = str(fileName)
                fileNameSplitted = fileName.replace(".jpg", "").split("-")
                if len(fileNameSplitted) <= 1:
                    continue
                face, created = "https://i.imgur.com/" + fileNameSplitted[0] + ".jpg", int(float(fileNameSplitted[1]))
                timeKeepings.append(TimeKeeping(fileName, face, created))
        return timeKeepings
