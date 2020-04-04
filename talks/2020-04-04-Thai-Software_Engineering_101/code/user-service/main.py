from src.config import Config
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
    ''' Greet the user '''

    return "User service is up and running"


@app.route('/users', methods=['GET', 'POST'])
def users():
    if request.method == 'POST':
        payload = request.get_json(force=True, silent=True)
        return process_users_post(payload)

    if request.method == 'GET':
        return process_users_get()


@app.route('/users/<username>', methods=['GET'])
def user(username):
    ''' Returns info about a specific user '''
    return process_user_get(username)


@app.route('/users/<username>/channel/<channel_id>', methods=['POST'])
def user_channel(username, channel_id):
    ''' Reference user to channel '''

    channel_json = get_channel(channel_id)
    channel = json.loads(channel_json)[0]
    response = mongo_handler.map_user_to_channel(username, channel)
    return json.dumps(response)


def get_channel(channel_id):
    try:
        req = requests.get(f"http://{Config.TALK_SERVICE}:5001/channels/{channel_id}")
    except requests.exceptions.ConnectionError:
        return "Service channel unavailable"
    return req.text


def process_users_post(payload):
    ''' Save user into database and return user id '''

    inserted_id = mongo_handler.persist_user(payload)
    return json.dumps(inserted_id, cls=JSONEncoder)


def process_users_get():
    ''' Return all users from database '''

    users = mongo_handler.get_users()
    json_users = json.dumps(users, cls=JSONEncoder)
    return json_users


def process_user_get(id):
    ''' Return user from database '''

    user = mongo_handler.get_user(id)
    return json.dumps(user, cls=JSONEncoder)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
