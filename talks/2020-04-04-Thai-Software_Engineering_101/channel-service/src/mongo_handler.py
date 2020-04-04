from .config import Config
from .utils.helper import Helper
from bson.objectid import ObjectId
import pymongo

class MongoHandler(object):

    def __init__(self):
        self._mongo_client = pymongo.MongoClient(host=Config.DB_HOST,
                                                 port=Config.DB_PORT, # necessary
                                                 username=Config.DB_USERNAME,
                                                 password=Config.DB_PASSWORD,
                                                 authSource=Config.DB_NAME,
                                                 authMechanism=Config.DB_AUTH)

    def _channel_collection(self):
        return self._mongo_client[Config.DB_NAME][Config.DB_COLLECTION_channels]

    def persist_channel(self, channel):
        if 'channel_id' not in channel:
            channel['channel_id'] = Helper.make_id()
        channel['_id'] = channel['channel_id']
        insert_one_result = self._channel_collection().insert_one(channel)
        return insert_one_result.inserted_id

    def get_channels(self):
        return list(self._channel_collection().find())
        
    def get_channel(self, id):
        return list(self._channel_collection().find({"_id": id}))