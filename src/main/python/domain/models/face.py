from src.main.python.domain.models.image import Image
import dlib

class Face:
    def __init__(self, x1, y1, x2, y2, image: Image):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.xCenter = (self.x1 + self.x2) // 2
        self.yCenter = (self.y1 + self.y2) // 2
        self.image = image

    def toDlibRectangle(self):
        return dlib.rectangle(
            left=self.x1,
            right=self.x2,
            top=self.y1,
            bottom=self.y2,
        )
    def getImage(self):
        return self.image