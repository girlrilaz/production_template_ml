Production-Ready Code Skeleton for Machine Learning and Deep Learning
==============================

## 1. Getting Started

To get started, first clone this repository to your machine. (see [docs] https://docs.github.com/en/free-pro-team@latest/github/creating-cloning-and-archiving-repositories/cloning-a-repository for help on this)

## 2. Create virtual environment

Creating virtual environment to ensure reproducability of this project.

##### OPTION 1: If you have Anaconda distribution installed in your machine, set up a virtual environment with 'conda'

In your terminal or command line, the command to create virtual environment with conda :

<i> conda create --name [replace_with_env_name] python=[replace_with_python_version] </i>

for this project, copy and paste the command below
```
conda create --name py38 python=3.8
```
##### OPTION 2: Set up a Python virtual environment with 'python-dotenv'

Make sure you have python-dotenv installed

```
pip install python-dotenv
```

Then create a the virtual environment for isolating this project on your machine using the following command:

```
python -m venv .venv
```

## 3. Activate the virtual environment

##### OPTION 1: Conda virtual environment

Windows or Mac - in the terminal or command line :

<i>conda activate [replace_with_env_name]</i>

```
conda activate py38
```

##### OPTION 2: dotenv virtual environment

Windows:
```
.\.venv\Scripts\activate
```
Mac:
```
source .venv/bin/activate
```

## 4. Install Python dependencies

```
pip install -r requirements.txt
```

If you need to extract a list of the depencies from this environment or another environment:

```
conda activate [env name]
pip freeze > requirements.txt
```

## 4. Gather the raw data

* for local dataset
Add your raw dataset(s) in data > raw folder

## 5. Set the paramaters in the config file 

There are a few configuration formats in this skeleton such as json, YAML and module, for this example use config in configs > module > config.py

## 6. Run the Model Suite

To run the model training on full dataset, use this:

```
python -m main full
```

To run the model training on a subset of the dataset, especially if you a large dataset and you want to train on only a subset and restrict training time, use this:

```
python -m main subset
```

## Unit Testing

The unit testing framework used for this project is [`unittest`](https://docs.python.org/3/library/unittest.html).
Tests are stored in the `tests/unittests` directory.
An alternative unit testing framework that can be used - Pytest

### Testing locally with make command

To run individual unit testing, check the commands in Makefile. For example, to run the API test, enter the following in your command line
```
python -m tests.unittests.MakeDataTests
```

To run all tests, 

```
python -m utils.tests
```

or

```
make tests
```

### Checking for testing coverage

Coverage measurement is typically used to gauge the effectiveness of tests. It can show which parts of your code are being exercised by tests, and which are not. To find out more please refer to : https://coverage.readthedocs.io/en/6.2/

To install coverage
```
pip install coverage
```

and to run it

```
coverage report
```

or

```
coverage html
```

for a prettier html version of the report.

## Dockerization

Make sure you have docker running in your desktop and have created a Dockerfile. To know more about docker, go here: https://docker-curriculum.com/

### Build docker image

```
docker build -t [image_name] . --load
```

### Run docker image

```
docker run -it -p 8080:8080 [image_name]
```

### Push docker image to your Docker Hub repository

```
docker tag [image_name] [docker_hub_profile]/[image_name]:[image_name]
docker login
docker push [docker_hub_profile]/[image_name]:[image_name]
```

### Project Organization

The project tree below is an example of a project organization:

------------

```
.
├── Dockerfile               <- Dockerfile to build docker image
├── Makefile                 <- Makefile to create run short cuts in CLI using make command
├── README.md                <- The top-level README for developers using this project.
├── app                      <- Working directory for model application
│   ├── app.py
│   ├── model_inferrer.py
│   ├── requirements.txt
│   ├── static
│   └── templates
├── configs                  <- Configureation files, can choose either JSON, YAML or Module format
│   ├── json
│   ├── module
│   └── yaml
├── data
│   ├── external             <- Data from third party sources.
│   ├── interim              <- Intermediate data that has been transformed.
│   ├── processed            <- The final, canonical data sets for modeling. 
│   └── raw                  <- The original, immutable data dump.
├── evaluation               <- Generated reports, graphics and figures to be used in reporting
├── executor                 <- Keep model executors such as trainer, evaluator and inferrer (predictor)
├── logs                     <- Where generated logfiles are kept
├── main.py                  <- Main script to run the entire code
├── models                   <- Model folder
│   ├── base_model.py        <- Base Model code building blocks script
│   ├── model.py             <- Model executors combined code script
│   └── saved_models         <- Folder to save trained models
├── notebooks                <- Jupyter notebooks. 
├── requirements.txt         <- The requirements file for reproducing the analysis environment, e.g.
├── tests                    <- where each individual unit test scripts are stored, can choose Pytest or Unittests
│   ├── pytest
│   └── unittests
└── utils                    <- folder to organize scripts, for example dataloader, visualize etc.

```
--------

## Other

General project setup steps and procedures - some may be an iterative process, for example logging and unit testing. So use this is a basic guide only

1. Configs
2. Dataloader
3. Data Pipelines / Processing
4. Model
5. Evaluation
6. Prediction
7. Inferrer
8. Logging
9. Unit testing
10. App
11. Unit testing
12. Dockerization
13. Model Serving

ALSO TRY:    
HEROKU - to deploy the model with some free templates    
FASTAPI - an alternative to Flask