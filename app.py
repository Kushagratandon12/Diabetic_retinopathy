from flask import Flask, request
from flask.helpers import url_for
from flask_pymongo import PyMongo
from components.core.database import save_users_images
from PIL import Image

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
    save_users_images(mongo, image, user_name)
    return 'Working', 200


# GET THE USER INFORMATION FROM MONGO_DB

@app.route('/get_information', methods=['POST'])
def show_user_diabetic():
    user_name = request.values['user_name']
    user = mongo.db.Diabetic.find_one_or_404({'username': user_name})
    filename = user['diab_image']
    return mongo.send_file(filename)


if __name__ == '__main__':
    app.run(debug=True)
