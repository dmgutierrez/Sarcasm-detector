# A Python Sarcasm Analyzer service

## Project

This project aims to perform a Sarcasm analysis based on Deep learning text embeddings computed from a large set of headlines from News. To do so, we have employed the NLP Library named as [**Flair**](https://github.com/flairNLP/flair) to both extract powerful text embeddings and train a **binary classifier** to distinguish News titles as sarcastic (positive class label) or non-sarcastic (negative class label).

Moreover, we have implemented a simple user interface in order to make easier the visualzation of the results. In this sense, we have used [**Flask**](https://flask.palletsprojects.com/en/1.1.x/#) to build the back-end of the application. 

## Dataset

In order to train the model, the *News Headlines Dataset* which contains a **large set of balanced sarcasm/Non-sarcasm headlines**. The Dataset can be downloaded from this [**Kaggle link**](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection).

We have included the dataset in both .csv and json format in the /resources folder.

## Models

Since the models are very large files, you need to **download the model** files from this [**link**](https://drive.google.com/uc?export=download&id=1aU-Cs7l0oQ2Ms2k4HSd7WD7ribzEBR3X
) and uncompress the **models.zip** file. This will generate a directory such as:

    .
    ├── models
        ├── sarcasm                 # Main Folder
        │   ├── best-model.pt       # Best model during training
        │   ├── final-model.pt      # Final trained model
        │   ├── loss.tsv            # Loss function values during training
        │   ├── test.tsv            # Test results during training
        │   ├── training.txt        # Training Description
        │   ├── weights.txt         # Model weights

## Models performance
The original dataset was split into three different subsets: **train, test and dev**. During the training process, the model is validated using the testing set. **Once the training process ends, we evalute the performance** of the **best model** using the remaining **dev subset**. The following table shows such performance results: 

|Class| F1-Score | Accuracy | Precision | Recall
| ------ | ------ | ------ | ------ | ------ |
| Sarcastic (positive) | 0.8932 | 0.8071 | 0.8877 | 0.8988
| Non-Sarcastic (negative) | 0.9017| 0.8210 | 0.9069 | 0.8966


## Set Up
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

## User Interface
- Main Page of the interface

![image](https://github.com/dmgutierrez/Sarcasm-detector/blob/master/images/main_page.PNG)

- Sarcasm Detector Result

![image](https://github.com/dmgutierrez/Sarcasm-detector/blob/master/images/result_page.PNG)

