from typing import List

import cv2
import imutils

from src.main.python.domain.models.image import Image

FlippedImage = Image

class ImageAugmenter:
    def augment(self, image: Image) -> List[Image]:
        pass
    def augments(self, images: List[Image]) -> List[Image]:
        pass
    def flips(self, images: List[Image]) -> List[FlippedImage]:
        pass

class ImageAugmenterImpl(ImageAugmenter):
    def augment(self, image: Image) -> List[Image]:

        return [image, cv2.flip(image, 1)]
        # images = []
        # width, height = image.shape[1], image.shape[0]
        # aQuaterX = width / 5.0
        # images.append(image)
        # images.append(cv2.flip(image, 1))
        # images.append(imutils.translate(image, aQuaterX, 0))
        # images.append(imutils.translate(image, -aQuaterX, 0))
        # return images

    def augments(self, images: List[Image]) -> List[Image]:
        augmentedImages = []
        for image in images:
            augmentedImages += self.augment(image)
        return augmentedImages

    def flips(self, images: List[Image]) -> List[FlippedImage]:
        return [cv2.flip(image, 1) for image in images]
