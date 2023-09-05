import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import train_test_split

"""
This is a research project for Machine Intelligence Systems as Full Sail University

goals are to Load the dataset 
clean any samples or features that need it
perform EDA
Choose a the right model for the data 
produce visualizations
do validation


Goal of the project is to predict if a passenger survived the titanic disaster
input: Pclass(Ticket Class: 1 = upper, 2 = middle, 3 = lower), Age, Parch(num parents), Sibsp(num sibling), Sex 
output: survived
"""

def load_data(filename):
    """Load data from CSV files."""
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    path = os.path.join(data_dir, filename)
    data = pd.read_csv(path, usecols=['Pclass', 'Age', 'Parch', 'SibSp', 'Sex', 'Survived'])
    df = pd.DataFrame(data)
    return df

def inspect_data(data):
    """Inspect the data."""
    print(data.describe())
    print(data.isnull().sum())


if __name__ == '__main__':
    filename = 'titanic_passengers.csv'
    df = load_data(filename)
    inspect_data(df)




