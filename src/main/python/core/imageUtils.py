import time

import cv2


class ImageUtils:
    @staticmethod
    def resizeImage(img, width, height):
        border_v = 0
        border_h = 0
        if (width / height) >= (img.shape[0] / img.shape[1]):
            border_v = int((((width / height) * img.shape[1]) - img.shape[0]) / 2)
        else:
            border_h = int((((height / width) * img.shape[0]) - img.shape[1]) / 2)
        img = cv2.copyMakeBorder(img, border_v, border_v, border_h, border_h, cv2.BORDER_CONSTANT, 0)
        return cv2.resize(img, (height, width))
    @staticmethod
    def increaseBrightness(image, value = 20):
        return image
        # start = time.time()
        # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # h, s, v = cv2.split(hsv)
        #
        # lim = 255 - value
        # v[v > lim] = 255
        # v[v <= lim] += value
        #
        # final_hsv = cv2.merge((h, s, v))
        # image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        # end = time.time()
        # print("Increase brightness in ", (end-start))
        # return image