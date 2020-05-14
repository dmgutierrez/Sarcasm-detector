# A Python Sarcasm Analyzer service

## Project

This project aims to perform a Sarcasm analysis based on Deep learning text embeddings computed from a large set of headlines from News. To do so, we have employed the NLP Library named as [**Flair**](https://github.com/flairNLP/flair) to both extract powerful text embeddings and train a **binary classifier** to distinguish News titles as sarcastic (positive class label) or non-sarcastic (negative class label).

Moreover, we have implemented a simple user interface in order to make easier the visualzation of the results. In this sense, we have used [**Flask**](https://flask.palletsprojects.com/en/1.1.x/#) to build the back-end of the application. 

## Dataset

In order to train the model, the *News Headlines Dataset* which contains a large set of balanced sarcasm/Non-sarcasm headlines. The Dataset can be downloaded from this [**Kaggle link**](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection).

We have included the dataset in both .csv and json format in the /resources folder.

## Requirements
All the requirements needed to run this project are included in the requirements.txt file. To make sure that you have all the dependencies, we recommend you to execute the following commands:
- Create a new Python environment i.e using Anaconda
```bash
conda create -n YOUR_ENV_NAME python=3.6
``` 
- Activate your environment
```bash
conda activate YOUR_ENV_NAME
```

- Clone or download the project
- Go to the parent directory of the project
- Install the required packages using the following command

```bash
pip install -r requirements.txt
```
- Configure the port and the host as you wish by creating two environment variables: 
    - **API_HOST** and **API_PORT**. By default, the host of the application is localhost and the port is 5000.

- Start the service using the following command
```bash
python app.py
```
- Go to http://your_host:your_port and enjoy!
