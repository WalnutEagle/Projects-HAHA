from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/quote')
def get_quote():
    response = requests.get('https://api.forismatic.com/api/1.0/?method=getQuote&format=json&lang=en')
    data = response.json()
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
