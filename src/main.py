import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.impute import KNNImputer

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
    print(data.info())

def convert_sex(df):
    """change Sex to int64 using oneHot"""
  #converting object to float for Sex
    oneshot = LabelBinarizer()
    res = oneshot.fit_transform(df['Sex'])
    pca = PCA(n_components=1)
    df['Sex_PCA'] = pca.fit_transform(res)
    df.drop(columns=['Sex'], axis=1, inplace=True)
    df['Sex'] = pca.fit_transform(res)
    df.drop(columns=['Sex_PCA'], axis=1, inplace=True)
    return df

def fill_age_knn(df):
    """fill in missing values for Age using knn."""

    # create a copy of the df
    df_imputed = df.copy()
    missing_age_data = df_imputed[df_imputed['Age'].isnull()]
    available_age_data = df_imputed[df_imputed['Age'].notnull()]

    # use knn imputer
    imputer = KNNImputer(n_neighbors=3)
    imputer.fit(available_age_data[['Age', 'Parch', 'SibSp']])
    imputed_data = imputer.transform(missing_age_data[['Age', 'Parch', 'SibSp']])
    imputed_data = np.round(imputed_data[:, 0], 2)
    df_imputed.loc[missing_age_data.index, 'Age'] = imputed_data

    return df_imputed

def correlation_plots(df, out_dir):
    """Create correlation plots."""
    # create a correlation matrix
    correlation_matrix = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(out_dir, 'correlation_matrix_input_features.png'))

    # pairplot of the data
    sns.pairplot(df)
    file_name = 'pairplot.png'
    path = os.path.join(out_dir, file_name)
    plt.savefig(path)
    plt.close(fig='all')


def train_dummy_regressor(X_train, y_train, X_test, y_test):
    """Train a Dummy Regressor and evaluate its score."""
    model = DummyRegressor(strategy='mean')
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    return model, score

"""
input: Pclass(Ticket Class: 1 = upper, 2 = middle, 3 = lower), Age, Parch(num parents), Sibsp(num sibling), Sex 
output: survived
"""
if __name__ == '__main__':
    filename = 'titanic_passengers.csv'
    out_dir = './output'
    df = load_data(filename)
    df = convert_sex(df)
    df = fill_age_knn(df)
    correlation_plots(df, out_dir)

    # split the data
    input_features = df['Pclass', 'Age', 'Parch', 'SibSp', 'Sex']
    target = df['Survived']




