# API Advertising Sales Prediction

## Description
With this API you can get a sales prediction based on TV, radio and newspaper investment data and also you can improve the model using new labeled data training whenever times you need.

## Considerations
This API and model is only for academic purposes so all data is not real.

## Data
- Investment (`X` or `predictors`):
  - `TV`: amount of investement used in TV media
  - `radio`: amount of investement used in radio media
  - `newspaper`: amount of investement used in newspaper media
- Sales (`y` or `labels`):
  - `sales`: amount of money as income of investements

![Advertising.csv original data](/images/AdvertisingData.png)


## Model
We used a simple `Lasso Regressor` just for creating a model that will be exported with `pickle` in order to be able to load it in the API when it's available. If not, API will create it.

## API
- `/`: base endpoint
- `/api/v1/predict`: use this for predicting `sales` using `TV`, `radio` and `newspaper` parameters in this order

![Sales predict example with Postman](/images/SalesPredict.png)

- `/api/v1/retrain`: use this for retraining model and receiving new Root Mean Squeared Error of the new model. It will write a new `pickle` file with the new model

![Retrain example with Postman](/images/Retrain.png)
