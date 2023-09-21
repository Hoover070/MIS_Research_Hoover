import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import ElasticNet
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, learning_curve, \
    validation_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import xgboost as xgb
import warnings

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
#usecols=['Pclass', 'Age', 'Parch', 'SibSp', 'Sex', 'Survived']) = 87/84/3 - default
#usecols=['Pclass', 'Age', 'Parch', 'SibSp', 'Survived'])  = 79/74/5 (no sex)
#usecols=['Pclass', 'Parch', 'SibSp','Sex', 'Survived']) = 80/80/0 (no age)
#secols=[ 'Age', 'Parch', 'SibSp', 'Sex', 'Survived']) = 85/82/3 (no pclass)
#usecols=['Pclass', 'Sex', 'Survived']) = 79/77/3 (no age, parch, sibsp)

def load_data():
    """Load data from Titanic CSV file"""
    filename = 'titanic_passengers.csv'
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

def class_balance(df, target):
    """Display a bar chart showing the class balance of a dataframe."""
    sns.countplot(x=target, data=df, order=sorted(df[target].unique()))
    plt.savefig(os.path.join(out_dir, f'class_balance_{target}.png'))



def convert_sex(df):
    """Sex change from obj to binary values"""
    df = df[df['Sex'] != '']
    df['Sex'] = df['Sex'].replace({'male': 0, 'female': 1})
    return df

def fill_age_knn(df):
    """fill in missing values for Age using KNNImputer"""

    df_imputed = df.copy()
    missing_age_data = df_imputed[df_imputed['Age'].isnull()]
    available_age_data = df_imputed[df_imputed['Age'].notnull()]

    imputer = KNNImputer(n_neighbors=3)
    imputer.fit(available_age_data[['Age', 'Parch', 'SibSp']])
    imputed_data = imputer.transform(missing_age_data[['Age', 'Parch', 'SibSp']])
    imputed_data = np.round(imputed_data[:, 0], 2)
    df_imputed.loc[missing_age_data.index, 'Age'] = imputed_data

    return df_imputed



def correlation_plots(df, out_dir):
    """Create correlation plots."""
    # create a correlation matrix
    correlation_matrix = df.corr()
    plt.figure(figsize=(15, 12))
    ax = sns.heatmap(correlation_matrix, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')


    # annotate the heatmap manually (annot is only annotating the first row and IDK how to fix it, so I'm doing it manually)
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


def optimize_knn_model(X_train, y_train):
    """Optimize the KNN model hyperparameters using GridSearchCV."""
    param_grid = {
        'n_neighbors': list(range(1, 100, 1)),
        'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }

    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_score = grid_search.best_score_
    best_params = grid_search.best_params_

    # Saving best parameters to a file
    with open("knn_best_params.txt", "w") as file:
        file.write(f"Best Hyperparameters: {best_params}\n")
        file.write(f"Best Score from GridSearchCV: {best_score}\n")

    return best_score, best_params

def knn_model_cv(X_train, y_train, X_test, y_test, k_folds=5, test_size=0.2, random_state=42, n_neighbors=3, metric='euclidean', weights='uniform', algorithm='auto'):

    #training KNN
    model = KNeighborsClassifier(n_neighbors=n_neighbors)  # Start with any k, say 3
    cv_scores = cross_val_score(model, X_train, y_train, cv=k_folds)
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.2f}, Std: {np.std(cv_scores):.2f}")
    model = model.fit(X_train, y_train)


    #prediction and testing
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    train_accuracy = model.score(X_train, y_train)

    difference = train_accuracy - test_accuracy
    print(f"Training Accuracy: {train_accuracy:.2f}")
    print(f"Testing Accuracy: {test_accuracy:.2f}")
    print(f"Train-Test Accuracy Difference: {difference:.2f}")

    if train_accuracy > test_accuracy and difference > 0.10:
        print(
            "The model might be overfitting because the training accuracy is significantly higher than the testing accuracy.")
    elif train_accuracy < 0.75 and test_accuracy < 0.75:  # Threshold of 0.6 is arbitrary; adjust as per your problem domain
        print("The model might be underfitting as both training and testing accuracies are low.")
    else:
        print("The model seems to be performing well and generalizing to new data.")

    return model



"""
input: Pclass(Ticket Class: 1 = upper, 2 = middle, 3 = lower), Age, Parch(num parents), Sibsp(num sibling), Sex 
output: survived
"""
if __name__ == '__main__':
    out_dir = '../output'

    df = load_data()
    df = convert_sex(df)
    df = fill_age_knn(df)
    df.info()

    for col in df.columns:
        class_balance(df, col)


    # split and transform the data for the model
    input_features = df.drop("Survived", axis=1)
    target = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(input_features, target, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # fit the baseline cv model
    model = knn_model_cv(X_train, y_train, X_test, y_test)

    print('Training a Dummy Classifier')
    dummy_model, dummy_score = train_dummy_classifier(X_train, y_train, X_test, y_test)
    print(f'Dummy Score: {dummy_score}')
    #
    # print('Optimizing KNN Model')
    # best_cv_score, best_params = optimize_knn_model(X_train, y_train)
    # print(f'Best optimized CV Score: {best_cv_score}')
    # print(f'Best optimized Parameters: {best_params}')
    #
    # # train a KNN model using optimized params
    # best_knn = KNeighborsClassifier(**best_params)
    # best_knn.fit(X_train, y_train)
    #
    # # Evaluate the KNN model on the test dataset
    # optimized_test_score = best_knn.score(X_test, y_test)
    # optimized_training_score = best_knn.score(X_train, y_train)
    # print(f'Optimized KNN Training Score: {optimized_training_score}')
    # print(f'Optimized KNN Test Score: {optimized_test_score}')

    cv_test_score = cross_val_score(model, X_test, y_test, cv=5)
    print(f'standard CV Test Score: {cv_test_score}')
    model_test_score = model.score(X_test, y_test)
    model_training_score = model.score(X_train, y_train)
    print(f'CV KNN Training Score: {model_training_score}')
    print(f'CV KNN Test Score: {model_test_score}')


    # train ElasticNet model
    # print('Training an ElasticNet Model')
    # elastic_net = ElasticNet()
    # elastic_net.fit(X_train, y_train)
    #
    # # evaluate the ElasticNet model
    # elastic_net_test_score = elastic_net.score(X_test, y_test)
    # elastic_net_training_score = elastic_net.score(X_train, y_train)
    # print(f'ElasticNet Training Score: {elastic_net_training_score}')
    # print(f'ElasticNet Test Score: {elastic_net_test_score}')

    #
    # correlation_plots(df, out_dir)
    #
    #
    #
    #
    # # learning curve
    # train_sizes, train_scores, valid_scores = learning_curve(model, input_features, target, cv=5)
    # plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
    # plt.plot(train_sizes, valid_scores.mean(axis=1), label='Validation score')
    # plt.xlabel("Training Set Size")
    # plt.ylabel("Accuracy Score")
    # plt.title("Learning Curve")
    # plt.legend()
    # plt.savefig(os.path.join(out_dir, 'learning_curve.png'))
    # plt.close()
    #
    # # validation curve
    # param_range = np.arange(1, 50, 2)
    # train_scores, valid_scores = validation_curve(model, input_features, target, param_range=param_range, cv=5, param_name='n_neighbors')
    # plt.plot(param_range, train_scores.mean(axis=1), label='Training score')
    # plt.plot(param_range, valid_scores.mean(axis=1), label='Validation score')
    # plt.xlabel("Number of Neighbors")
    # plt.ylabel("Accuracy Score")
    # plt.title("Validation Curve")
    # plt.legend()
    # plt.savefig(os.path.join(out_dir, 'validation_curve.png'))
    # plt.close()
    #
    # # confusion matrix
    # conf_mat = confusion_matrix(y_test, model.predict(X_test))
    #
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
    #             xticklabels=['Not Survived', 'Survived'],
    #             yticklabels=['Not Survived', 'Survived'])
    # plt.ylabel('Actual')
    # plt.xlabel('Predicted')
    # plt.title("Confusion Matrix")
    # plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
    # plt.close()
    #
    #
    # # precision recall curve
    # precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
    #
    # plt.figure()
    # plt.plot(recall, precision, color='b', lw=1)
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall Curve')
    # plt.savefig(os.path.join(out_dir, 'precision_recall_curve.png'))
    #
    #
    #
    #
    #
    #
