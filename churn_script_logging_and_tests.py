import os
import logging
import pytest
import numpy as np
import pandas as pd
# import churn_library as cls
from churn_library import import_data, churn_column, perform_eda, perform_feature_engineering, encoder_helper, train_models, generate_predictions, plot_classification_results, feature_importance_plot, classification_report_image

logging.basicConfig(
    filename= r'C:\Users\hp\Coding World\MLOps\Customer_Churn_Prediction_End_to_End_Project\logs\churn_library.log', #os.path.join('logs', 'churn_library.log'),
    level=logging.INFO,
    filemode='a',
    format='%(name)s - %(levelname)s - %(message)s')

# Print and save the filename
check = os.path.join('logs', 'churn_library.log')
print(f"Logging to file: {check}")


@pytest.fixture(scope="module")
def path():
    return r"datasets\bank_data.csv"


@pytest.fixture(scope="module")
def import_data_fixture(path):
  file_path = path
  return import_data(file_path)


@pytest.fixture(scope="module")
def churn_column_fixture(import_data_fixture):
    return churn_column(import_data_fixture)


@pytest.fixture(scope="module")
def encoder_helper_fixture(churn_column_fixture):
    df = churn_column_fixture
    cat_columns = df.select_dtypes(exclude='number').columns.tolist()
    category_lst = list(set(cat_columns) - set(['Attrition_Flag']))
    return encoder_helper(churn_column_fixture, category_lst)


@pytest.fixture(scope="module")
def perform_feature_engineering_fixture(encoder_helper_fixture):
    df = encoder_helper_fixture
    x_train, x_test, y_train, y_test = perform_feature_engineering(df)
    return x_train, x_test, y_train, y_test

@pytest.fixture(scope="module")
def generate_predictions_fixture(perform_feature_engineering_fixture):
    x_train, x_test, y_train, y_test = perform_feature_engineering_fixture
    y_train_preds, y_test_preds = generate_predictions(x_train, x_test, 'rfc_model')
    return y_train_preds, y_test_preds


def test_import(import_data_fixture):
    '''
    test data import - this example is completed for you
    to assist with the other test functions
    '''
    try:
        # file_path = path
        df = import_data_fixture
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_churn_column(import_data_fixture):
    try:
        df = churn_column(import_data_fixture)
        logging.info("SUCCESS: Churn Column added. Logger dictionary: " + str(logging.getLogger().manager.loggerDict))

        assert 'Churn' in list(df.columns)
    except AssertionError as err:
        logging.error("Failed to add column!")
        raise err
    except KeyError as err:
        logging.error("Attrition_Flag column was not found!")
        raise err


def test_eda(churn_column_fixture):
    '''
        test perform eda function
    '''
    # Ensure that the 'images/eda' directory exists
    os.makedirs('images/eda', exist_ok=True)

    try:
        perform_eda(churn_column_fixture)
        logging.info("SUCCESS: EDA plots were successfully plotted")
        # Check if the required image files are generated
        assert os.path.isfile(
            'images/eda/churn_histogram.png'), "Could not save churn histogram plot"
        assert os.path.isfile(
            'images/eda/customer_age.png'), "could not save customer age plot"
        assert os.path.isfile(
            'images/eda/marital_status.png'), "could not save marital status plot"
        assert os.path.isfile(
            'images/eda/Total_Trans_Ct.png'), "Could not save Total_Trans_Ct plot"
        assert os.path.isfile(
            'images/eda/corr_heatmap.png'), "Could not save correlation heatmap plot"
    except AssertionError as err:
        logging.error("FAILED:{%s}", str(err))
        raise err


def test_encoder_helper(churn_column_fixture):
    '''
    test encoder helper
    '''
    df = churn_column_fixture
    dfcopy = df.copy()
    cat_columns = df.select_dtypes(exclude='number').columns.tolist()
    category_lst = list(set(cat_columns) - set(['Attrition_Flag']))

    try:
        new_df = encoder_helper(dfcopy, category_lst)
        logging.info("SUCCESS: The encoding was successful")

        # Check input types
        assert isinstance(
            df, pd.DataFrame), "Input 'df' must be a pandas DataFrame."

        # Check for NaN values in categorical columns
        assert not df[category_lst].isnull().any().any(
        ), "Categorical columns should not contain missing values."

        # Check to confirm that the right number of columns have been added
        # and/or dropped
        assert len(
            df.columns) == len(
            new_df.columns), "The number of Columns before and after must be the same"

    except AssertionError as err:
        logging.error("FAILED: {%s}", str(err))
        raise RuntimeError("FAILED: {%s}", str(err)) from err


def test_perform_feature_engineering(encoder_helper_fixture):
    '''
    Test perform_feature_engineering
    '''
    try:
        # Get the DataFrame from encoder_helper_fixture
        df = encoder_helper_fixture

        # Set the response column
        response = 'Churn'

        # Check input types
        assert isinstance(
            df, pd.DataFrame), "Input 'df' must be a pandas DataFrame."

        # Check if the response column exists
        assert response in df.columns, f"Response column '{response}' found not in DataFrame."

        # Perform feature engineering
        x_train, x_test, y_train, y_test = perform_feature_engineering(df)

        # Logging success message
        logging.info("SUCCESS: Features to train the model was successfully created")

        # Columns that should be dropped
        dropped_columns = [
            'Attrition_Flag',
            'CLIENTNUM',
            'Churn',
            'Unnamed: 0']

        # Check if dropped columns still exist in x_train
        assert all(
            col not in x_train.columns for col in dropped_columns
        ), "Dropped columns still exist in x_train."

        # Check if dropped columns still exist in x_test
        assert all(
            col not in x_test.columns for col in dropped_columns
        ), "Dropped columns still exist in x_test."

        # Test size for train-test split
        test_size = 0.3

        # Check if the total number of samples changed after train-test split
        assert x_train.shape[0] + x_test.shape[0] == len(
            df), "Total number of samples did not change after train-test split."
        assert y_train.shape[0] + y_test.shape[0] == len(
            df), "Total number of samples did not change after train-test split."

        # Check if the number of features are same in train and testing data
        # after train-test split
        assert x_train.shape[1] == x_test.shape[1
                                                ], "Total number of samples did not change after train-test split."

        # Check if the test size is equal to the specified test size
        assert round(x_test.shape[0] / (x_train.shape[0] + x_test.shape[0]),
                     1) == test_size, f"Test size is not equal to {test_size}."

    except AssertionError as err:
        logging.error("FAILED: {%s}", str(err))
        raise RuntimeError("FAILED: {%s}", str(err)) from err


def test_train_models(perform_feature_engineering_fixture):
    '''
    test train_models function
    '''
    x_train, x_test, y_train, y_test = perform_feature_engineering_fixture
    try:
        assert isinstance(x_train, (np.ndarray, pd.DataFrame)
                          ), "x_train must be a numpy array or pandas DataFrame."
        assert isinstance(x_test, (np.ndarray, pd.DataFrame)
                          ), "x_test must be a numpy array or pandas DataFrame."
        assert isinstance(y_train, (np.ndarray, pd.Series)
                          ), "y_train must be a numpy array or pandas Series."
        assert isinstance(y_test, (np.ndarray, pd.Series)
                          ), "y_test must be a numpy array or pandas Series."

        # Call the function
        rfc, lrc = train_models(x_train, x_test, y_train, y_test)

        # Test if the models are fitted
        assert hasattr(
            rfc, 'classes_'), "RandomForestClassifier is not fitted."
        assert hasattr(lrc, 'classes_'), "LogisticRegression is not fitted."

        # Test if the ROC curve plots are generated and saved
        assert os.path.isfile(os.path.join('images', 'results', 'lr_model_ROC.png')
                              ), "ROC curve plot for Logistic Regression not found."
        assert os.path.isfile(
            os.path.join(
                'images', 'results', 'rf_model_ROC.png')
        ), "ROC curve plot for Random Forest not found."

        # Test if the models are saved
        assert os.path.isfile(
            os.path.join(
                'models', 'rfc_model.pkl')), "RandomForestClassifier model not saved."
        assert os.path.isfile(
            os.path.join(
                'models', 'logistic_model.pkl')), "LogisticRegression model not saved."

    except AssertionError as err:
        logging.error("FAILED: {%s}", str(err))
        raise RuntimeError("FAILED: {%s}", str(err)) from err

    except Exception as err:
        logging.error(
            "FAILED: Model training failed with error - {%s}",
            str(err))

def test_generate_predictions(perform_feature_engineering_fixture, model_name):
    '''
    test generate_predictions function
    '''
    x_train, x_test, _, _ = perform_feature_engineering_fixture
    model_name = model_name
    try:
        y_train_preds, y_test_preds = generate_predictions(x_train, x_test, model_name)
        assert len(y_train_preds) == len(x_train), "There might be a problem with the prediction. The Length of the target is different from the length of the training data"
        assert len(y_test_preds) == len(x_test), "There might be problem with the prediction. The Length of the target is different from the length of the test data"
        
    except (FileNotFoundError, AttributeError, ValueError, TypeError) as err:
        logging.error("Generate_Predictions Function Failed: {%s}", str(err))
        raise RuntimeError("FAILED: {%s}", str(err)) from err
    
    except AssertionError as err:
        logging.error("Generate_Predictions Function Error: {%s}", str(err))


def test_plot_classification_results(perform_feature_engineering_fixture, generate_predictions_fixture, model_name):
    '''
    test plot_classification_results function
    '''
    _,_, y_train, y_test = perform_feature_engineering_fixture
    y_train_preds, y_test_preds  = generate_predictions_fixture
    model_name = model_name

    try:
    # Test successful case
        plot_classification_results(model_name, y_train_preds, y_test_preds, y_train, y_test)

        # Make assertions for successful case
        assert os.path.exists(os.path.join('images','results', f"{model_name}_results.png")), "The classification results image was either not created or could not be saved in the path"

    except (ValueError, TypeError) as err:
        logging.error("Classification Plot Function Failed: {%s}", str(err))
        raise RuntimeError("FAILED: {%s}", str(err)) from err
    
    except AssertionError as err:
        logging.error("Classification Plot Function Error: {%s}", str(err))


def test_classification_report_image(perform_feature_engineering_fixture, model_name):
    '''
    test classification_report_image function
    '''
    x_train,x_test, y_train, y_test = perform_feature_engineering_fixture
    model_name = model_name

    try:
 
        classification_report_image(y_train, y_test, x_train, x_test)

        # Make assertions for successful case
        image_path_rfc = os.path.join('images', 'results', 'rfc_model_result.png')
        image_path_lr = os.path.join('images', 'results', 'logistic_model_result.png')
        assert os.path.exists(image_path_rfc)
        assert os.path.exists(image_path_lr)

    except (FileNotFoundError, ValueError, TypeError) as err:
        logging.error("Classification Report Function Failed: {%s}", str(err))
        raise RuntimeError("FAILED: {%s}", str(err)) from err
    
    except AssertionError as err:
        logging.error("Classification Report Function Error: {%s}", str(err))


def test_feature_importance_plot(encoder_helper_fixture, model_name):
    '''
    test feature importance plot function
    '''

    model = model_name
    x_data = encoder_helper_fixture
    output_pth = os.path.join('images', 'results', f'{model}_importances.png')

    try:
        feature_importance_plot(model, x_data, output_pth)
        # Make assertions for successful case
        assert os.path.exists(os.path.join('images','results', f"{model_name}_importances.png")), "The feature importance plot was either not created or could not be saved in the path"

    except (FileNotFoundError, ValueError, TypeError) as err:
        logging.error("Feature Importance Plot Function Failed: {%s}", str(err))
        # raise RuntimeError("FAILED: {%s}", str(err)) from err
    
    except AssertionError as err:
        logging.error("Feature Importance Plot Function Error: {%s}", str(err))







# if __name__ == "__main__":
    # Run the test using pytest
    # import_data(r"datasets/bank_data.csv")
    # test_import(path)
# 	test_churn_column(import_data_fixture)
# 	test_eda(churn_column_fixture)

if __name__ == "__main__":
    # Run the test using pytest
    pytest.main(["-v"])
