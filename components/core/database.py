import os
from bson.objectid import ObjectId
from bson.objectid import ObjectId
from dotenv.main import load_dotenv
from flask_pymongo import MongoClient


# MongoDB Details Saved In ENV
load_dotenv('../utils/.env')
MONGO_DB_CREDENTIAL = os.getenv('MONGO_DB_CREDENTIAL')


client = MongoClient(MONGO_DB_CREDENTIAL)
db = client.get_database('myFirstDatabase')
diabetic = db.Diabetic['username']
files = db.fs.files
chunks = db.fs.chunks


def save_users_images(mongo, image, username):
    mongo.save_file(image.filename, image)
    mongo.db.Diabetic.insert(
        {'username': username, 'filename': image.filename, 'diabetic_result': 'None'})
    return True


def get_user_data(mongo, username):
    filename = (mongo.db.Diabetic.find_one_or_404(
        {'username': username}))['filename']
    return filename


def get_user_image_id(mongo, username):
    files_id = list(files.find({'filename': get_user_data(mongo, username)}))
    image_id = (files_id[0])['_id']
    result = list(chunks.find(
        {"files_id": ObjectId(image_id)}))
    image_bytes = result[0]['data']
    return image_bytes


def get_user_id(mongo, username):
    files_id = list(files.find({'filename': get_user_data(mongo, username)}))
    image_id = (files_id[0])['_id']
    return image_id
