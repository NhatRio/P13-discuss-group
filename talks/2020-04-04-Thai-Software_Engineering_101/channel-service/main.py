from src.utils.json_encoder import JSONEncoder
from src.mongo_handler import MongoHandler
from flask import Flask, jsonify, request, make_response
import requests
import os
import json

app = Flask(__name__)

mongo_handler = MongoHandler()


@app.route("/", methods=['GET'])
def hello():
    ''' Greet the channel '''

    return "Channel service is up and running"


@app.route('/channels', methods=['GET', 'POST'])
def channels():
    if request.method == 'POST':
        payload = request.get_json(force=True, silent=True)
        return process_post_channel(payload)

    if request.method == 'GET':
        return process_get_channels()


@app.route('/channels/<channel_id>', methods=['GET'])
def channel(channel_id):
    ''' Returns info about a specific channel '''
    return process_get_channel(channel_id)


def process_post_channel(payload):
    ''' Save channel into database and return channel id '''

    inserted_id = mongo_handler.persist_channel(payload)
    print(f'process_post_channel: {inserted_id}')
    return json.dumps(inserted_id, cls=JSONEncoder)


def process_get_channels():
    ''' Return all channels from database '''

    channels = mongo_handler.get_channels()
    return json.dumps(channels, cls=JSONEncoder)


def process_get_channel(id):
    ''' Return channel from database '''

    channel = mongo_handler.get_channel(id)
    return json.dumps(channel, cls=JSONEncoder)


if __name__ == '__main__':
    app.run(port=5001, debug=True)