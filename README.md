trip-duration
==============================

A sample ML project to build an end-to-end working app for the NYC taxi trip duration challenge.

We are creating a service, that will take a few inputs and return the trip duration
The service will be hosted/deployed on internet and It will work only with a specific input and trip duration
Here, User visit the url /url/predict?data and the Server in return give the trip duration to the User.

# setup vscode and anaconda
    if not installed

# create env and activate env and install pip and check if python and pip is installed or not.
    open Anaconda Prompt
    conda create -n ds python=3.10 (create env in not created)
    conda activate ds
    check for pip, dvc and git and python (if not then install) (git must be on the global system and dvc must be on the base env of anaconda while the rest must be env specific)
    conda install pip
    pip --version
    python --version
    dvc --version (if not then check with base env)
    git --version

# Create the project using CookieCutter template
    pip install cookiecutter (if not installed)
    cookiecutter -c v1 https://github.com/drivendata/cookiecutter-data-science
    You've downloaded C:\Users\Vivek\.cookiecutters\cookiecutter-data-science before. Is it okay to delete and re-download it? [y/n] (y): y
        project_name (project_name): trip-duration
        repo_name (trip-duration): trip-duration
        author_name (Your name (or your organization/company/team)): Vivek Kumar
        description (A short description of the project.): A sample ML project to build an end-to-end working app.
        Select open_source_license
            1 - MIT
            2 - BSD-3-Clause
            3 - No license file
            Choose from [1/2/3] (1): 3
        s3_bucket ([OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')): buckets/trip-duration-nyc-taxi
        aws_profile (default): 
        Select python_interpreter
            1 - python3
            2 - python
            Choose from [1/2] (1): 1
    NOTE: steps to create/get s3_bucket name
        Login to aws console
        visit https://ap-south-1.console.aws.amazon.com/s3/home?region=ap-south-1 or s3
        Click on Create Bucket
        give a unique name. Here, i have given trip-duration-nyc-taxi
        make sure Block all public access is on
        the required s3_bucket name will be like buckets/name-of-the-bucket. Here, i have given trip-duration-nyc-taxi.

# Collect the data
download the data from the https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data
move the data train and test csv to the data/raw/ after unzipping

# Initiate a repo
    cd project_name  (Here, trip-duration in this case)
    code . (opening the vscode)
        NOTE: select the interpreter if not selected using Ctrl + Shift + p
    git init (Initializing the git repo)

# Track the data and model folder with DVC
    pip install dvc (to install the dvc in the env if not installed)
    click on the DVC button of DVC extension integration
    if says says "DVC not found" or not green
    Then they select the interpreter (or DVC path) of the the env (if dvc is installed in the the env)
    Finally, you see "DVC (auto)" at the bottom status bar ✅
    dvc init
    dvc add data/ (adding data folder in the dvc tracking)
    git add data.dvc .gitignore (To track the changes with git)
    dvc add models/ (same for models/)
    git add .gitignore models.dvc

# commit data changes with your code exclusively (if data is changed manually then it must be commit to git along with dvc)
    **Git hooks**
    **Checkout**: Automatically runs dvc checkout after git checkout to update workspace data.
    **Commit/Reproduce**: Automatically checks DVC status before git commit to remind using dvc commit or dvc repro if needed.
    **Push**: Automatically runs dvc push before git push to upload tracked data to remote storage.
    dvc install

# EDA using the notebook
    Create a notebook in the notebooks folder for EDA and find the best model and its parameters

# code convert from notebook to scr/
    Here, we have to go through the EDA notebook and convert the code into .py (s) inside scr folder
    if required create make_dataset.py to create a train and test split. (take inspiration from creditcard project)

# create stages to run with DVC
    Here, we have created the dvc.yaml file to run the stages on running dvc repro or dvc exp run
    dvc repro (once the yaml files and scr files have been created)
    git add dvc.lock (the above command will create a dvc.lock file so add it to git tracking list)

# add code to create the app.py
    create app.py and write code in it using uvicorn/streamlit

# in get_dataset.py, add connectors to get data globally

# Create a Branch for the best model using hyperparameter tunning and logging the best model to mlflow
    when dvc repro will run hyperparameter tunning will happen inside the train_model.py and we will get best model
    save the model and load the model in training then train it on all the data
    model and model performance tracking using mlflow on cloud is necessary for actual mlops
    i will create a push_model.py in place of predict_model.py i.e. src/models/push_model.py
    send the best model to the aws to the ECR or S3 (model registry) manually 
    then from S3, we will read the model to our code
    parallely dvc is controlling the model
    dvc repro
    git add dvc.lock models.dvc
    dvc push

# mlflow setup
    Open your terminal and run:
        pip install mlflow
    To verify installation:
        mlflow --version
    Run the MLflow Tracking UI (local)
        mlflow ui or mlflow server
    By default, this opens at:
        http://127.0.0.1:5000
    Visit the link to access the tracking page and click on Run's 'Model' column in the Experiments, to get the details of the particular best model
    To register the model
        click on Register model
        Select a model (initially you have to click on create new model or when you want to create a new model)
        If we select the model that was previously selected/created, a new version (v1, v2 and so on) of the model will be registered.
    To restore previous best model
        ######### search with chatgpt
        
    Press Ctrl + C to exit the mlflow ui
    Ctrl + C
NOTE: when the mlflow is running on the EC2, we can pull the model directly from the mlflow to retraining etc.

# connecting aws cli in local computer
    For Windows, Download AWS CLI v2 from the below link. Once downloaded, installed it via the installer:
        https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-windows.html
    To verify installation:
        aws --version
    You should see something like:
        aws-cli/2.15.3 Python/3.x.x Windows/10 exe/x86_64
    Set Up AWS Credentials
        aws configure
    It will ask for
        AWS Access Key ID:     <YOUR_ACCESS_KEY>
        AWS Secret Access Key: <YOUR_SECRET_KEY>
        Default region name:   us-east-1  # or your region
        Default output format: json       # or text or table
    Verify Setup (list your buckets if credentials are working)
        aws s3 ls

# push your code to github repo
    mkdir TEMP (this will create a TEMP folder that will work as server, temporarily)
    dvc remote add -d localtemp ./TEMP (once the folder is created, then add remote server to dvc)
    dvc push (to push the your DVC-tracked data)
    add TEMP/ inside .gitignore (so git will ignore the TEMP/ folder)

# Connect AWS with the GitHub
    go to your repo inside github then settings then secrets and variables and lastly add secrets
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_REGION
    AWS_ECR_LOGIN_URI
    ECR_REPOSITORY_NAME
    GH_PERSONAL_ACCESS_TOKEN

# Create a self-hosted runner
    go to EC2 in AWS and create a instance
    click on instance ID once the instance is created
    click on connect then click on connect again. A terminal will open
    Now, go to the your project repo then settings then Actions then Runners
    Then Click on New self-hosted runner then click on linux. Run the given commands in the terminal

        mkdir actions-runner && cd actions-runner

        curl -o actions-runner-linux-x64-2.323.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.323.0/actions-runner-linux-x64-2.323.0.tar.gz

        echo "0dbc9bf5a58620fc52cb6cc0448abcca964a8d74b5f39773b7afcad9ab691e19  actions-runner-linux-x64-2.323.0.tar.gz" | shasum -a 256 -c

        tar xzf ./actions-runner-linux-x64-2.323.0.tar.gz

        ./config.sh --url https://github.com/mrvivekkumar7171/trip-duration --token A5ORRA6I2BKHNU7GXCHPJXTIAE4ZA

        Enter the name of the runner group to add this runner to: [press Enter for Default] (skipped)
        Enter the name of runner: [press Enter for ip-172-31-6-56] trip-runner (here, i have named it trip-runner)
        This runner will have the following labels: 'self-hosted', 'Linux', 'X64' 
        Enter any additional labels (ex. label-1,label-2): [press Enter to skip] (skipped)
    lastly you will successfull message 
        √ Runner successfully added
        √ Runner connection is good

        ./run.sh (to start the runner)

NOTE: every project must have a unique self-hosted runner and if wants more than one runner in a EC2 instance then change the name of the folder actions-runner in the above command

# Dockerfile and it's dev-requirements.txt creation
    create the Dockerfile with model.joblib, app, requirements.txt and scr folder for build_feature function in an app folder
    installing requirements and running the app.py

# Using CI with the best model Branch
    creating the ci/cd pipeline
## In CI,
    nothing done in CI
## Build and push image to ECR
    Install Utilities
    Configure/Login AWS credentials
    Build, tag, and push the containerize image to image hub (Amazon ECR)
    report via cml - model metrics
## In CD,
    using self-hosted runner by the name trip-runner
    Configure/Login AWS credentials
    Pull latest images on EC2
    Delete Previous Container
    Run/Deploy Docker Image to serve users via the app.py

NOTE: To successfully run the CD, the self-hosted runner must be active/idle.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
