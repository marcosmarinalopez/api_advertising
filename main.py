from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return """
    <h1>APP para calcular las ventas a partir de los gastos de marketing</h1>    
    """

@app.route('/api/v1/predict', methods=['GET'])
def predict():
    predicted_sales = -1

    try:        
        tv = float(request.args['TV'])
        radio = float(request.args['radio'])
        newspaper =  float(request.args['newspaper'])
        #sales = float(request.args['sales'])

        model_advertising= pickle.load(open('model.pkl', 'rb'))

        data_for_prediction = [tv, radio, newspaper]

        prediction = model_advertising.predict([data_for_prediction])
        predicted_sales = prediction[0]

    except Exception as ex:
        predicted_sales = "Bad parameters given. " + str(ex)
    
    finally:
        return jsonify({'Predicted Sales: ':predicted_sales})

if __name__ == '__main__':
    app.run(debug = True, port=5000)