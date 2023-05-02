import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
import pickle
import os

# Class for training model with original dataset even with new data
class Trainer:
    
    def __init__(self, id):
        self.id = id

    # Train model with original data and new one if available
    def train_model(self, new_data):
        isOk = True    
        rmse = -1
        try:
            os.chdir(os.path.dirname(__file__))

            data = pd.read_csv('data/Advertising.csv', index_col=0)

            if new_data != None: 
                df_new_data = pd.read_csv(new_data, index_col=0)
                data = pd.concat([data, df_new_data])

            X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['sales']),
                                                                data['sales'],
                                                                test_size = 0.20,
                                                                random_state=42)

            model = Lasso(alpha=6000)
            model.fit(X_train, y_train)

            pickle.dump(model, open('model.pkl', 'wb'))
            rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
            print("RMSE: ", rmse)

        except Exception as ex:
            print("Exception in train method: "+ str(ex))
            isOk = False
            rmse = ex
        
        return (isOk,rmse)