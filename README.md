# Spaceship Titanic project. Kaggle competition
## Author: Andrii Zhurba
The relevant project is located in src directory. 
Exploratory data analysis is in EDA.ipynb file,
while the modeling part is in Modeling.ipynb file.

## Description
It's 2912 year... The Spaceship Titanic is an 
interstellar passenger liner launched a month ago. 
With almost 13,000 passengers on board, the vessel 
set out on its maiden voyage transporting emigrants
from our solar system to three newly habitable 
exoplanets orbiting nearby stars. While rounding 
Alpha Centauri en route to its first destination—the 
torrid 55 Cancri E—the unwary Spaceship Titanic 
collided with a spacetime anomaly hidden within 
a dust cloud. Sadly, it met a similar fate as its
namesake from 1000 years before. Though the ship
stayed intact, almost half of the passengers 
were transported to an alternate dimension!

To help rescue crews and retrieve the 
lost passengers, we are challenged to predict 
which passengers were transported by the anomaly 
using records recovered from the spaceship’s 
damaged computer system.

## Objective:
The primary goal of this project is to develop a
machine learning model that accurately predicts 
whether a passenger was transported by the anomaly.
The model's performance will be measured by its
accuracy, with a minimum acceptable score of 0.79.

We will deploy our best-performing model using FastAPI on Google Cloud Run. Docker has been used to containerize the application

## Results:
After thorough examination, **CatBoost** model has 
been chosen as the most effective for our problem. 
The final model's performance on the test set is 
almost 80% which means our goal of having at least 79% 
score is achieved.

## Data
The dataset for this project is provided by the Kaggle competition "Spaceship Titanic."
It includes the following files:

* train.csv: Contains the training data, which includes features about the passengers and the target variable indicating if they were transported.
* test.csv: Contains the test data without the target variable, which the model will predict.
* submission.csv: A sample submission file in the correct format, which will be used to submit predictions to Kaggle.

The dataset can be downloaded from the competition page on Kaggle: https://www.kaggle.com/competitions/spaceship-titanic/overview

To run the notebook, install the following dependencies:
```commandline
pip install -r requirements.txt
```
