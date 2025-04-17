# Use an official Python runtime as a parent image
# change the python version as per need in the docker image
FROM python:3.8-slim

# Set the working directory to /app or say creating a directory called app
WORKDIR /app

# Copy the required files and directory into the container at /app
COPY app.py /app/app.py
COPY model.joblib /app/model.joblib
# to run the build_features real time
COPY src/ /app/src/
COPY requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
# --no-cache-dir prevents pip from storing the downloaded packages, reducing the image size by hundreds of MB
RUN pip install --no-cache-dir -r requirements.txt

# NOT worked
# Copy files from S3 inside docker
# RUN mkdir /app/models
# RUN aws s3 cp s3://creditcard-project/models/model.joblib /app/models/model.joblib

EXPOSE 8080

# Run app.py when the container launches
CMD ["python", "app.py"]