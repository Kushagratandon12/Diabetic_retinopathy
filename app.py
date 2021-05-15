import os
from dotenv import load_dotenv
from flask import Flask, request
from flask_pymongo import PyMongo
# IMPORTING FROM MY FUNCTIONS
from components.model.model_predict import model_pred
from components.core.database import save_users_images, get_user_data, get_user_image_id, get_user_id
load_dotenv()
# MongoDB Details Saved In ENV
MONGO_DB_CREDENTIAL = os.getenv('MONGO_DB_CREDENTIAL')

app = Flask(__name__)
app.config['MONGO_URI'] = MONGO_DB_CREDENTIAL
mongo = PyMongo(app)

# SAVE THE USER INFORMATION TO MONGO_DB


@app.route('/', methods=['POST'])
def save_user_diabetic():
    if request.method == 'GET':
        return 'Send Your Post Request Here'
    image = request.files['image']
    user_name = request.values['user_name']
    save_users_images(mongo, image, user_name)
    id = get_user_id(mongo, user_name)
    return 'User Registration Completed with Image Id {}'.format(id), 200


# GET THE USER INFORMATION FROM MONGO_DB

@app.route('/get_information', methods=['POST'])
def show_user_info():
    user_name = request.values['user_name']
    filename = get_user_data(mongo, user_name)
    return mongo.send_file(filename)


@app.route('/predict', methods=['POST'])
def predict_info():
    user_name = request.values['user_name']
    filename = get_user_data(mongo, user_name)
    images_byte = get_user_image_id(mongo, user_name)
    res = model_pred(images_byte)
    # print(res)
    return 'The Stage Of Diabetic You Are At is {}'.format(res), 200


if __name__ == '__main__':
    app.run(debug=True)
