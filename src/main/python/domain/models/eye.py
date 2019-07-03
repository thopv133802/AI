from typing import Tuple

import numpy as np

class Eye:
    def __init__(self, x1, y1, x2, y2):
        if x1 > x2:
            tempX = x1
            x1 = x2
            x2 = tempX
            tempY = y1
            y1 = y2
            y2 = tempY
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.xCenter = (self.x1 + self.x2) // 2
        self.yCenter = (self.y1 + self.y2) // 2
    def translate(self, x, y):
        self.x1 += x
        self.x2 += x
        self.y1 += y
        self.y2 += y

        self.xCenter += x
        self.yCenter += y

class Eyes:
    def __init__(self, eye1: Eye, eye2: Eye):
        if (eye1.x1 < eye2.x1):
            self.leftEye = eye1
            self.rightEye = eye2
        else:
            self.leftEye = eye2
            self.rightEye = eye1
    def calculateAngle(self):
        rightEyeYCenter, rightEyeXCenter, leftEyeYCenter, leftEyeXCenter = self.rightEye.yCenter, self.rightEye.xCenter, self.leftEye.yCenter, self.leftEye.xCenter

        dY = rightEyeYCenter - leftEyeYCenter
        dX = rightEyeXCenter - leftEyeXCenter
        angle = np.degrees(np.arctan2(dY, dX))

        if(angle > 90 or angle < -90):
            return 0

        return angle

    def calculateCenter(self):
        rightEyeYCenter, rightEyeXCenter, leftEyeYCenter, leftEyeXCenter = self.rightEye.yCenter, self.rightEye.xCenter, self.leftEye.yCenter, self.leftEye.xCenter
        return (rightEyeXCenter + leftEyeXCenter) / 2, (rightEyeYCenter + leftEyeYCenter) / 2