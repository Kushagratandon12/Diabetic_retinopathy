from typing import Type
from bson.objectid import ObjectId
from flask_pymongo import MongoClient
from bson.objectid import ObjectId

# MongoDb Details
client = MongoClient(
    'mongodb+srv://Kushagra:samkush#@cluster0.p9ece.mongodb.net/myFirstDatabase?retryWrites=true&w=majority')
db = client.get_database('myFirstDatabase')
diabetic = db.Diabetic['username']
files = db.fs.files
chunks = db.fs.chunks

# print(diabetic)


def save_users_images(mongo, image, username):
    mongo.save_file(image.filename, image)
    mongo.db.Diabetic.insert(
        {'username': username, 'filename': image.filename, 'diabetic_result': 'None'})
    return True


# def check_dublicate_user(username):
#     user_valid = list(diabetic.find({'username:' }))
#     print(user_valid)
#     return 'Hi'


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


