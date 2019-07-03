import json

import numpy

from src.main.python.domain.services.memberService import Member, TimeKeeping


class MyJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Member):
            return {"userId": obj.userId, "faces": [{"id": face.id, "src": face.src} for face in obj.faces]}
        if isinstance(obj, TimeKeeping):
            return {
                "id": obj.id,
                "face": obj.face,
                "created": obj.created
            }
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)