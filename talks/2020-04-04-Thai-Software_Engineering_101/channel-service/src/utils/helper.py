import uuid


class Helper:
    
    @staticmethod
    def make_id():
        return uuid.uuid4().hex
