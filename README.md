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
- [`GET`] `/`: base endpoint
- [`GET`] `/api/v1/predict`: use this for predicting `sales` using `TV`, `radio` and `newspaper` parameters in this order

![Sales predict example with Postman](/images/SalesPredict.png)

- [`POST`] `/api/v1/retrain`: use this for retraining model and receiving new `Root Mean Squeared Error` of the new model uploading a new `.csv` file with the same structure as `Advertisement.csv` original file. Also, this new file must be `retrain_file.csv` for being able to retrain the model. It will write a new `pickle` file with the new model. In the below example, we use Postman for uploading `retrain_file.csv` in the `Body` section and in `form-data` subsection, selecting this file and the path. 

![Retrain example with Postman](/images/Retrain.png)
