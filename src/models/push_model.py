import boto3 # sdk for AWS services
import shutil # for file operations
from botocore.exceptions import NoCredentialsError

def upload_to_s3(local_file_path, bucket_name, s3_file_path):
    # Create/Initialize an S3 client using boto3
    s3 = boto3.client('s3')

    try:
        # Upload the file
        s3.upload_file(local_file_path, bucket_name, s3_file_path)
        print(f"File uploaded successfully to {bucket_name}/{s3_file_path}")
    except FileNotFoundError:
        # if file not present in the path
        print(f"The file {local_file_path} was not found.")
    except NoCredentialsError:
        # if the credentials are not available
        print("Credentials not available.")

# Example usage
local_model_path = 'models/model.joblib'
s3_bucket_name = 'trip-duration-nyc-taxi' # replace with your actual S3 bucket name
s3_file_path = 'models/model.joblib' 
# so the path will become s3://trip-duration-nyc-taxi/models/model.joblib

# uploading to s3 bucket
upload_to_s3(local_model_path, s3_bucket_name, s3_file_path)
# here, i am keeping a copy of the best model in the root folder for the docker creation
# i am unable to securely give access to the s3 bucket in the docker file.
shutil.copy(local_model_path, 'model.joblib')