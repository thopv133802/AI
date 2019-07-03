import imutils


class ImageOptimizer:
    def optimize(self, image):
        pass
    def optimizes(self, images):
        pass


class ImageOptimizerImpl(ImageOptimizer):
    def optimize(self, image):
        return image
    def optimizes(self, images):
        return [self.optimize(image) for image in images]