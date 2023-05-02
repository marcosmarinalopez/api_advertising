from flask import Flask

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    return "Predicción aquí"

if __name__ == '__main__':
    app.run(port=5000)