import json
from typing import List
import numpy as np


class Embeddings:
    def toString(self) -> str:
        return json.dumps([str(element) for element in list(self)])
