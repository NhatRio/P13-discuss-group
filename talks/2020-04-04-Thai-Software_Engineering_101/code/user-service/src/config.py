import logging


class Config(object):
    DB_HOST = 'database'
    DB_PORT = 27017
    DB_USERNAME = 'test'
    DB_PASSWORD = 'test@2020'
    DB_NAME = 'test'
    DB_AUTH = 'SCRAM-SHA-256'
    DB_COLLECTION_USERS = 'users'
    DB_COLLECTION_channels = 'channels'
    TALK_SERVICE = 'channel-service'
