import os
import logging
import pytest
# import churn_library as cls
from churn_library import *

logging.basicConfig(
    filename= os.path.join('logs', 'churn_library.log'),
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

@pytest.fixture(scope="module")
def path():
    return r"datasets/bank_data.csv"

@pytest.fixture(scope="module")
def import_data_fixture(path):
    return import_data(path)

@pytest.fixture(scope="module")
def churn_column_fixture(import_data_fixture):
    return Churn_Column(import_data_fixture)

@pytest.fixture(scope="module")
def encoder_helper_fixture(churn_column_fixture):
    df = churn_column_fixture
    cat_columns = df.select_dtypes(exclude='number').columns.tolist()
    category_lst = list(set(cat_columns) - set(['Attrition_Flag']))
    return encoder_helper(churn_column_fixture, category_lst)

@pytest.fixture(scope="module")
def perform_feature_engineering_fixture(encoder_helper_fixture):
    df = encoder_helper_fixture
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    return X_train, X_test, y_train, y_test

def test_import(path):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = import_data(path)
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_data: The file wasn't found")
		raise err
	
	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err

def test_churn_column(import_data_fixture):
	try:
		df = Churn_Column(import_data_fixture)
		logging.info("SUCCESS: Churn Column added successfully")
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
        assert os.path.isfile('images/eda/churn_histogram.png'), "Could not make churn histogram plot"
        assert os.path.isfile('images/eda/customer_age.png'), "could not make customer age plot"
        assert os.path.isfile('images/eda/marital_status.png'), "could not make marital status plot"
        assert os.path.isfile('images/eda/Total_Trans_Ct.png'), "Could not make Total_Trans_Ct plot"
        assert os.path.isfile('images/eda/corr_heatmap.png'), "Could not make correlation heatmap plot"
    except AssertionError as err:
        logging.error(f"FAILED: {str(err)}")
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
        assert isinstance(df, pd.DataFrame), "Input 'df' must be a pandas DataFrame."

    	# Check for NaN values in categorical columns
        assert not df[category_lst].isnull().any().any(), "Categorical columns should not contain missing values."

		# Check to confirm that the right number of columns have been added and/or dropped
        assert len(df.columns) == len(new_df.columns), "The number of Columns before and after must be the same"
   
    except AssertionError as err:
        logging.error(f"FAILED: {str(err)}")
        raise RuntimeError(f"FAILED: {str(err)}")

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
        assert isinstance(df, pd.DataFrame), "Input 'df' must be a pandas DataFrame."

        # Check if the response column exists
        assert response in df.columns, f"Response column '{response}' not found in DataFrame."

        # Perform feature engineering
        X_train, X_test, y_train, y_test = perform_feature_engineering(df)

        # Columns that should be dropped
        dropped_columns = ['Attrition_Flag', 'CLIENTNUM', 'Churn', 'Unnamed: 0']

        # Check if dropped columns still exist in X_train
        assert all(col not in X_train.columns for col in dropped_columns), "Dropped columns still exist in X_train."

        # Check if dropped columns still exist in X_test
        assert all(col not in X_test.columns for col in dropped_columns), "Dropped columns still exist in X_test."

        # Test size for train-test split
        test_size = 0.3

        # Check if the total number of samples changed after train-test split
        assert X_train.shape[0] + X_test.shape[0] == len(df), "Total number of samples did not change after train-test split."
        assert y_train.shape[0] + y_test.shape[0] == len(df), "Total number of samples did not change after train-test split."

        # Check if the number of features are same in train and testing data after train-test split
        assert X_train.shape[1] == X_test.shape[1], "Total number of samples did not change after train-test split."
        
        # Check if the test size is equal to the specified test size
        assert round(X_test.shape[0] / (X_train.shape[0] + X_test.shape[0]), 1) == test_size, f"Test size is not equal to {test_size}."

    except AssertionError as err:
        logging.error(f"FAILED: {str(err)}")
        raise RuntimeError(f"FAILED: {str(err)}")


def test_train_models(perform_feature_engineering_fixture):
    '''
    test train_models
    '''
    X_train, X_test, y_train, y_test = perform_feature_engineering_fixture
    try:
        assert isinstance(X_train, (np.ndarray, pd.DataFrame)), "X_train must be a numpy array or pandas DataFrame."
        assert isinstance(X_test, (np.ndarray, pd.DataFrame)), "X_test must be a numpy array or pandas DataFrame."
        assert isinstance(y_train, (np.ndarray, pd.Series)), "y_train must be a numpy array or pandas Series."
        assert isinstance(y_test, (np.ndarray, pd.Series)), "y_test must be a numpy array or pandas Series."
        
        # Call the function
        rfc, lrc = train_models(X_train, X_test, y_train, y_test)

        # Test if the models are fitted
        assert hasattr(rfc, 'classes_'), "RandomForestClassifier is not fitted."
        assert hasattr(lrc, 'classes_'), "LogisticRegression is not fitted."

        # Test if the ROC curve plots are generated and saved
        assert os.path.isfile(os.path.join('images', 'results', 'lr_model_ROC.png')), "ROC curve plot for Logistic Regression not found."
        assert os.path.isfile(os.path.join('images', 'results', 'rf_model_ROC.png')), "ROC curve plot for Random Forest not found."

        # Test if the models are saved
        assert os.path.isfile(os.path.join('models', 'rfc_model.pkl')), "RandomForestClassifier model not saved."
        assert os.path.isfile(os.path.join('models', 'logistic_model.pkl')), "LogisticRegression model not saved."

    except AssertionError as err:
        logging.error(f"FAILED: {str(err)}")
        raise RuntimeError(f"FAILED: {str(err)}")
    
    except Exception as e:
        logging.error(f"FAILED: Model training failed with error - {str(e)}")
        


# if __name__ == "__main__":
#     # Run the test using pytest
#     # import_data(r"datasets/bank_data.csv")
# 	test_import(path)
# 	test_churn_column(import_data_fixture)
# 	test_eda(churn_column_fixture)

# if __name__ == "__main__":
#     # Run the test using pytest
#     pytest.main(["-v"])









