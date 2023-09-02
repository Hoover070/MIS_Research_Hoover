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

"""

def load_data():
    """Load data from CSV files."""
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    data = pd.read_csv(os.path.join(data_dir, 'titanic_passengers.csv'))
    return data

