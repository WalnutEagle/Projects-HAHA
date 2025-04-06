from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

def get_weather(city_name, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather_info = {
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed'],
            'visibility': data['visibility'] / 1000,
            'weather_description': data['weather'][0]['description'],
            'weather_icon': data['weather'][0]['icon']
        }
        return weather_info
    else:
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_weather', methods=['POST'])
def fetch_weather():
    city_name = request.form['city']
    api_key = "18f5cc74f647c0cd721dfbbd158f365e" 
    weather_info = get_weather(city_name, api_key)
    if weather_info:
        return jsonify(weather_info)
    else:
        return jsonify({'error': 'Error fetching weather data.'}), 500

if __name__ == "__main__":
    app.run(debug=True)
