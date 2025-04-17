# if we have 10 input features, let we have created/derived 20 features out of it.
# The resposibility of creating these 20 features in real time when we send the request to the model after it is deployed online.
# The problem is that when the model is deployed on the server. It also have only 10 inputs features but you model is trained on the 20 
# created/derived features so we have to create 20 created/derived features in real time from the 10 input features provided at the server.

# take train pipeline, transform all the features and write them to processed folder
# similarly, to test pipeline.

# run when dvc repro is called and when the app.py get user input

    # if we change the build_features.py and we run the dvc repro then dvc will know that the build_features.py is changed 
    # and it will run the build_features.py and the train_model.py again.

import pandas as pd
import numpy as np
import pathlib
from sklearn.model_selection import train_test_split
from feature_definitions import feature_build


def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df

# def split_data(df, test_split, seed):
#     # Split the dataset into train and test sets
#     train, test = train_test_split(df, test_size=test_split, random_state=seed) 
#     return train, test

def save_data(train, test, output_path):
    # Save the split datasets to the specified output path 
    pathlib.Path(output_path).mkdir (parents=True, exist_ok=True) 
    train.to_csv(output_path + '/train.csv', index=False) 
    test.to_csv(output_path + '/test.csv', index=False)

if __name__ == '__main__':
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    train_path = home_dir.as_posix() + '/data/raw/train.csv'
    test_path = home_dir.as_posix() + '/data/raw/test.csv'

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    output_path = home_dir.as_posix() + '/data/processed'

    train_data = feature_build(train_data, 'train-data')
    test_data = feature_build(test_data, 'test-data')

    save_data(train_data, test_data, output_path)