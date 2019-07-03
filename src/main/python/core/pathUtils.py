import os

_resourcePath = os.path.join("/root", "Projects", "ai", "src", "main", "resources")
# _resourcePath = os.path.join("C:\\", "Users", "thopv", "PycharmProjects", "ai", "src", "main", "resources")
_faceDetectorModelPath =os.path.join( _resourcePath, "faceDetector", "opencv_face_detector_uint8.pb")
_faceDetectorConfigPath = os.path.join(_resourcePath, "faceDetector", "opencv_face_detector.pbtxt")
_shapePredictorFilePath = os.path.join(_resourcePath,  "faceDetector", "shape_predictor_5_face_landmarks.dat")
_faceClassifierFolderPath = os.path.join(_resourcePath, "faceClassifier")
_faceClassifierFilePath = os.path.join(_faceClassifierFolderPath, "classifier.pkl")
_faceNetModelFilePath = os.path.join(_resourcePath, "facenet", "20180402-114759.pb")
_userRepositoryFilePath = os.path.join(_resourcePath, "repository", "users")
_imageFolderPath = os.path.join(_resourcePath, "images")
_faceLivenessFolderPath = os.path.join(_resourcePath, "faceLivenessDetector")



class PathUtils:
    @staticmethod
    def getFaceLivenessFolderPath():
        return _faceLivenessFolderPath
    @staticmethod
    def getFaceClassifierFilePath():
        return _faceClassifierFilePath
    @staticmethod
    def getFaceClassifierFolderPath():
        return _faceClassifierFolderPath
    @staticmethod
    def getResourcePath():
        return _resourcePath

    @staticmethod
    def getFaceDetectorModelPath():
        return _faceDetectorModelPath

    @staticmethod
    def getFaceDetectorConfigPath():
        return _faceDetectorConfigPath

    @staticmethod
    def getShapePredictorFilePath():
        return _shapePredictorFilePath

    @staticmethod
    def join(*path):
        return os.path.join(*path)

    @staticmethod
    def getFaceNetModelFilePath():
        return _faceNetModelFilePath

    @staticmethod
    def getUserRepositoryFilePath():
        return _userRepositoryFilePath

    @staticmethod
    def getImagesFolderPath():
        return  _imageFolderPath