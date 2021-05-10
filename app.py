from flask import Flask , request  , render_templates

# client = pymongo.MongoClient("mongodb+srv://<username>:<password>@cluster0.p9ece.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")

app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def diabetic_model_wel():
    if request.method == 'GET':
        return 'Send The Chatbot Post Request Here'
    return 'Working' ,200