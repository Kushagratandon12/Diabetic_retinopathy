from flask import Flask, request
from flask_pymongo import PyMongo
# IMPORTING FROM MY FUNCTIONS
from components.core.database import save_users_images, get_user_data, get_user_image_id
from components.model.model_predict import model_pred

app = Flask(__name__)
app.config['MONGO_URI'] = 'mongodb+srv://Kushagra:samkush#@cluster0.p9ece.mongodb.net/myFirstDatabase?retryWrites=true&w=majority'
mongo = PyMongo(app)


# SAVE THE USER INFORMATION TO MONGO_DB


@app.route('/', methods=['POST'])
def save_user_diabetic():
    if request.method == 'GET':
        return 'Send Your Post Request Here'
    image = request.files['image']
    user_name = request.values['user_name']
    # check if the user name is already registerd or not  -> database
    # flag_user = check_dublicate_user(mongo, user_name)
    # print(flag_user)
    save_users_images(mongo, image, user_name)
    return 'User Registration Completed.', 200


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
