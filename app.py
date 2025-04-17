# app.py is the backend of the FastAPI application and it handles the prediction logic and serves the API endpoints.

# main.py
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel
import pandas as pd
# to convert/create the 15 features from the below 8 input features in real time
# NOTE: if the feature creation must be easy or fast otherwise it will take more time thus not practical in real world scenarios.
from src.features.feature_definitions import feature_build

app = FastAPI()

class PredictionInput(BaseModel):
    # Define the input parameters required for making predictions
    vendor_id: float
    pickup_datetime: float
    passenger_count: float
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: float

# Load the pre-trained RandomForest model from main directory as unable to configure s3 bucket in docker securely
model_path = "model.joblib"
model = load(model_path)

@app.get("/")
def home():
    return "Working fine"

@app.post("/predict")
def predict(input_data: PredictionInput):
    # Extract features from input_data and make predictions using the loaded model
    features = {
            'vendor_id': input_data.vendor_id,
            'pickup_datetime': input_data.pickup_datetime,
            'passenger_count': input_data.passenger_count,
            'pickup_longitude': input_data.pickup_longitude,
            'pickup_latitude': input_data.pickup_latitude,
            'dropoff_longitude': input_data.dropoff_longitude,
            'dropoff_latitude': input_data.dropoff_latitude,
            'store_and_fwd_flag': input_data.store_and_fwd_flag
}   
    # creating dataframe of one raw for every request
    features = pd.DataFrame(features, index=[0])
    # creating the 15 features from the 8 input features
    features = feature_build(features, 'prod')
    # making the prediction
    prediction = model.predict(features)[0].item()
    # Return the prediction
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)


# we can use ipython in terminal to get the input dictionary's data types
# temp = dict(zip(df.columns, df.iloc[1,:]))
# for k, v in temp.items():
#     print(f"{k}: float")

# to get features names
# for k, v in temp.items():
#     print(f"input_data.{k},")

# to get the a test input dictionary (paste it inside the https://127.0.0.1:8080/docs)
# import json
# json.dumps(temp)

# ctrl + d to exit the ipython shell


# Note: make sure you don't include the target feature (here, it is trip_duration) in the input data dictionary.
# so you have to remove it from the input data dictionary before using it for prediction.


# http://127.0.0.1:8080 or http://localhost:8080 : It is the localhost and the port number where the FastAPI application is running. When you're developing/testing locally(Privately) on your laptop, you usually use it. Loopback address (localhost) Only accessible from your own machine via browser. No one else can connect to it.

# http://0.0.0.0:8080 : When you're deploying app.py on a Cloud Server like AWS EC2, GCP etc. This allows external access (like from your browser on another computer) from internet via: http://<your-aws-ec2-public-ip>:8080. All interfaces The app will listen on all available IPs on the machine, so it's accessible from outside your system too.

# NOTE: If you used 127.0.0.1 on AWS, the app would still run, but only AWS itself could access it — you couldn’t open it in your browser from your home.