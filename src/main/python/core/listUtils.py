from typing import List

import numpy


class ListUtils:
    @staticmethod
    def createListWithSize(size: int) -> List:
        return [None] * size
    @staticmethod
    def argmax(values) -> int:
        return numpy.argmax(values).reshape(-1)[0]
    @staticmethod
    def max(values) -> int:
        return numpy.max(values).reshape(-1)[0]
    @staticmethod
    def sum(values):
        return numpy.sum(values)
