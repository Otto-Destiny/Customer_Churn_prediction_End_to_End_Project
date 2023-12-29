# library doc string
'''
This Script contains all functions to run the data science model pipeline
for the Customer Churn PRediction Project
'''

# import libraries
import os
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# @pytest.fixture(scope="module")


def import_data(pth):
    '''
    returns dataframe for the csv found at pth
    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    return df

def churn_column(df):
    """
    Create a Churn column in the DataFrame based on the 'Attrition_Flag' column.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame.

    Returns:
    pandas.DataFrame: The DataFrame with the newly created 'Churn' column.
    """
    # Create a Churn Column from Attrition_Flag
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # Identify categorical columns
    cat_columns = df.select_dtypes(exclude='number').columns

    # Visualize  distribution of the Churn Column
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig('images\\eda\\churn_histogram.png')

    # Plot histogram of Customer Age
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig('images\\eda\\customer_age.png')

    # Plot histogram of normalized Marriage Status
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('images\\eda\\marital_status.png')

    # Plot histogram of Total_Trans_Ct
    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('images\\eda\\Total_Trans_Ct.png')

    # Plot Correlation heatmap
    quant_columns = list(set(df.columns.tolist()) - set(cat_columns))
    plt.figure(figsize=(20, 10))
    sns.heatmap(
        df[quant_columns].corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.savefig('images\\eda\\corr_heatmap.png')


def encoder_helper(df, category_lst, response='Churn'):
    '''
    Helper function to turn each categorical column into a new column with
    proportion of churn for each category.

    Parameters:
        df: pandas DataFrame
        category_lst: list of columns that contain categorical features
        response: string of response name [optional argument that could 
        be used for naming variables or index y column]

    Returns:
        df: pandas DataFrame with new columns
    '''
    # print("Columns before encoding:", df.columns, len(df.columns))  # Add
    # this line for debugging
    for i in category_lst:
        df[i + '_' + response] = df.groupby(i)[response].transform('mean')

    # print("Columns before drop:", df.columns, len(df.columns))  # Add this
    # line for debugging
    df.drop(columns=category_lst, inplace=True)
    # print("Columns after drop:", df.columns, len(df.columns))  # Add this
    # line for debugging

    return df


def perform_feature_engineering(df, response='Churn'):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that 
              could be used for naming variables or index y column]

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y = df[response]
    x = df.drop(
        columns=[
            'Attrition_Flag',
            'CLIENTNUM',
            'Churn',
            'Unnamed: 0'],
        axis=1)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                x_train,
                                x_test):
    '''
    produces classification report for training and testing results 
    and stores report as image in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    y_train_preds_rf, y_test_preds_rf = generate_predictions(x_train, x_test, 'rfc_model')
    y_train_preds_lr, y_test_preds_lr = generate_predictions(x_train, x_test, 'logistic_model')

    # plot random forest model results and save
    plot_classification_results(
        'Random Forest',
        y_train_preds_rf,
        y_test_preds_rf,
        y_train,
        y_test)
    # plt.savefig(os.path.join('images', 'results', 'rfc_output.png'))

    # plot logistic regression model results and save
    plot_classification_results(
        'Logistic Regression',
        y_train_preds_lr,
        y_test_preds_lr,
        y_train,
        y_test)
    # plt.savefig(os.path.join('images', 'results', 'lr_output.png'))


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    model_select = joblib.load(f'./models/{model}.pkl')

    # Calculate feature importances
    importances = model_select.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth + '/feature_importances.png')
    plt.close()


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              rfc: Fitted Randome Forest Model
              lrc: Fitted Logistic Regression Model
    '''
    # Initialize models
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000, n_jobs=12)

    # Set up hyperparameters for grid search
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['log2', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # Fit model and Grid search
    cv_rfc = GridSearchCV(
        estimator=rfc,
        param_grid=param_grid,
        cv=5,
        n_jobs=12)
    cv_rfc.fit(x_train, y_train)
    lrc.fit(x_train, y_train)

    # Plot ROC curve for Logistic Regression model
    lrc_plot = RocCurveDisplay.from_estimator(lrc, x_test, y_test)
    plt.savefig(os.path.join('images', 'results', 'lr_model_ROC.png'))
    plt.close()

    # Plot ROC curve for both Logistic Regression and Random Forest
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = RocCurveDisplay.from_estimator(
        rfc, x_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    rfc_disp.plot(ax=ax, alpha=0.8)
    plt.savefig(os.path.join('images', 'results', 'rf_model_ROC.png'))
    plt.close()

    # save best models
    joblib.dump(
        cv_rfc.best_estimator_,
        os.path.join(
            'models',
            'rfc_model.pkl'))
    joblib.dump(lrc, os.path.join('models', 'logistic_model.pkl'))

    return rfc, lrc


def plot_classification_results(
        model_name,
        train_preds,
        test_preds,
        y_train,
        y_test):
    """
    Plot and save classification results for a given model.

    Parameters:
    - model_name (str): Name of the model for labeling.
    - train_preds (numpy array): Predictions on the training data.
    - test_preds (numpy array): Predictions on the testing data.
    - y_train (numpy array): True labels for the training data.
    - y_test (numpy array): True labels for the testing data.

    Returns:
    - None
    """
    # Set up the figure size
    plt.rc('figure', figsize=(6, 7))

    # Plot and save the results
    plt.text(0.01, 1.25, f'{model_name} Train', {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_test, test_preds)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, f'{model_name} Test', {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, train_preds)), {
            'fontsize': 10}, fontproperties='monospace')

    # Turn off axis
    plt.axis('off')

    # # Save the plot
    plt.savefig(os.path.join('images', 'results', f'{model_name}.pkl'))

    # Close the plot to free up resources
    plt.close()


def generate_predictions(x_train, x_test, model_name):
    """
    Generate predictions using a trained model.

    Parameters:
    - x_train (numpy array): Features of the training data.
    - x_test (numpy array): Features of the testing data.
    - model_name (str): Name of the saved model file without the extension.

    Returns:
    - y_train_preds (numpy array): Predictions on the training data.
    - y_test_preds (numpy array): Predictions on the testing data.
    """

    # Load the trained model from the specified file
    model = joblib.load(f'models/{model_name}.pkl')

    # Predict with the loaded model
    y_train_preds = model.predict(x_train)
    y_test_preds = model.predict(x_test)

    return y_train_preds, y_test_preds

# @pytest.fixture(scope="module")


if __name__ == "__main__":
    import_data(r"datasets/bank_data.csv")
