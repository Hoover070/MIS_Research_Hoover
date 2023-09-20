import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split,  GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import xgboost as xgb
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


#added to get rid of the old python warning for seaborn
warnings.simplefilter(action='ignore', category=FutureWarning)

"""
This is a research project for Machine Intelligence Systems as Full Sail University

goals are to Load the dataset 
clean any samples or features that need it
perform EDA - determin if the data is linear or non-linear, and if it needs to be scaled
Choose a the right model for the data 
produce visualizations
do validation

We will then use a classification model to predict if a passenger survived the titanic



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
    """change Sex to binary values"""
    df = df[df['Sex'] != '']
    df['Sex'] = df['Sex'].replace({'male': 0, 'female': 1})
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

def scale_data(df):
    """Scale the data."""
    # create a copy of the df
    df_scaled = df.copy()

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df_scaled)

    return pd.DataFrame(scaled_values, columns=df.columns)

def correlation_plots(df, out_dir):
    """Create correlation plots."""
    # create a correlation matrix
    correlation_matrix = df.corr()
    plt.figure(figsize=(15, 12))
    ax = sns.heatmap(correlation_matrix, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')


    # Annotate the heatmap manually
    for i in range(correlation_matrix.shape[0]):
        for j in range(correlation_matrix.shape[1]):
            ax.text(j + 0.5, i + 0.5, f'{correlation_matrix.iloc[i, j]:.2f}',
                    ha='center', va='center', color='black')

    plt.savefig(os.path.join(out_dir, 'correlation_matrix_input_features.png'))

    # pairplot of the data
    sns.pairplot(df)
    file_name = 'pairplot.png'
    path = os.path.join(out_dir, file_name)
    plt.savefig(path)
    plt.close(fig='all')


def train_dummy_classifier(X_train, y_train, X_test, y_test):
    """Train a Dummy Classifier and evaluate its score."""
    model = DummyClassifier()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return model, score


def optimize_model(X_train, y_train):
    """Optimize the learning rate for the XGBoost Classifier."""
    param_grid = {
        'learning_rate': [0.05, 0.1, 0.2, 0.3, .5, .7, 1],
        'max_depth': [3, 6, 9],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
    }

    model = xgb.XGBClassifier(objective='binary:logistic', tree_method='hist')

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_score = grid_search.best_score_

    return grid_search.best_params_, best_score


def xgboost_classifier(X_train, y_train, X_test, y_test, rounds=100, early_stopping_rounds=10, best_params=None):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'eval_metric': 'error'
    }
    if best_params:
        params.update(best_params)

    evals_result = {}
    bst = xgb.train(params, dtrain, num_boost_round=rounds, evals=[(dvalid, 'valid')],
                    early_stopping_rounds=early_stopping_rounds, evals_result=evals_result)

    best_round = bst.best_iteration
    best_score = evals_result['valid']['error'][best_round]

    print(f"Best Round: {best_round}. Best Score: {best_score}")
    return bst

def validate_model(model, X_test, y_test):
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)
    y_pred_binary = np.round(y_pred).astype(int)

    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    conf_matrix = confusion_matrix(y_test, y_pred_binary)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)


"""
input: Pclass(Ticket Class: 1 = upper, 2 = middle, 3 = lower), Age, Parch(num parents), Sibsp(num sibling), Sex 
output: survived
"""
if __name__ == '__main__':
    filename = 'titanic_passengers.csv'
    out_dir = '../output'

    """Parameters"""
    num_rounds = 1000
    early_stopping_rounds = 10 #number of iterations WITHOUT improvement before the model stops
    df = load_data(filename)
    df = convert_sex(df)
    df = fill_age_knn(df)
    #correlation_plots(df, out_dir)
    #df = scale_data(df)

    # split the data for the 1st model - XGB Classifier to predict survival of a passenger
    df['FamilySize'] = df['Parch'] + df['SibSp']
    input_features = df[['Pclass', 'Age', 'FamilySize', 'Sex']]
    who_survived = df['Survived'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(input_features, who_survived, test_size=0.2, random_state=42)

    print('Training a Dummy Classifier')
    model, score = train_dummy_classifier(X_train, y_train, X_test, y_test)
    print(f'Dummy Score (if you cant beat this go back to the drawing board: {score}')

    print('Training a XGBoost Classifier')
    best_params, best_score = optimize_model(X_train, y_train)
    print("Best Parameters:", best_params)
    bst = xgboost_classifier(X_train, y_train, X_test, y_test, best_params=best_params)
    print('Done Training')

    validate_model(bst, X_test, y_test)
    bst.save_model(os.path.join(out_dir, 'xgb_classifier.model'))
    print('Model Saved')


    # see which features were most important in the decision
    # results are surprising, Sex was least important with Age being most important.
    xgb.plot_importance(bst)
    plt.savefig(os.path.join(out_dir, 'feature_importance.png'))
    plt.close(fig='all')






