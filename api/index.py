from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return 'aakash is awesome!'

@app.route('/about')
def about():
    return 'About aakash'