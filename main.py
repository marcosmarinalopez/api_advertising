import os
from flask import Flask, request, jsonify
import pickle
from train import Trainer
import pandas as pd

# New Flask instance
app = Flask(__name__)

# Define a folder for uploading files
UPLOAD_FOLDER = 'temp'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

# Default entrypoint
@app.route('/', methods=['GET'])
def home():
    return """
    <h1>APP for predicting advertisement sales</h1>    
    """

# It predicts the sales based in 3 parameters. If model is not created yet it will.
@app.route('/api/v1/predict', methods=['GET'])
def predict():
    predicted_sales = -1
    isOk = True # operation control boolean

    try:
        # parameters        
        tv = float(request.args['TV'])
        radio = float(request.args['radio'])
        newspaper =  float(request.args['newspaper'])
        
        # create the model if not created
        if os.path.exists('./model.pkl') == False:
            trainer = Trainer("trainer")
            isOk = trainer.train_model(None)
            isOk = isOk[0]

        # if model is available, let's predict
        if isOk:
            # get the model
            model_advertising= pickle.load(open('model.pkl', 'rb'))
            # get parameter for model
            data_for_prediction = [tv, radio, newspaper]
            # predict
            prediction = model_advertising.predict([data_for_prediction])
            predicted_sales = prediction[0]
        else:
            predicted_sales = "Error when training..."

    except Exception as ex:
        predicted_sales = "Bad parameters given. " + str(ex)
    
    finally:
        return jsonify({'Predicted Sales: ':predicted_sales})


# Retrain the model using new data within csv with the same structure than original train dataset
# There is a retrain_file.csv used for testing
@app.route('/api/v1/retrain', methods=['POST'])
def retrain():
    isOk = True
    rmse = -1

    try:        
        # currently, the file for retraining must be retrain_file.csv
        if 'retrain_file' not in request.files:
            rmse = "retrain_file.csv missing. Load a valid file."
                
        # get the file
        file = request.files['retrain_file']
        # save it into temporary folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)        

        # train again the model with original and new data
        trainer = Trainer("retrainer")
        results = trainer.train_model(file_path)
        # 
        isOk  = results[0]
        rmse = results[1]

    except Exception as ex:
        rmse = "Error when retraining. " + str(ex)
    
    finally:
        return jsonify({'Operation succesfully done?: ':str(isOk), 'RMSE: ': rmse})



if __name__ == '__main__':
    app.run(debug = True, port=5000)