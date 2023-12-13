import os
import logging
import pytest
# import churn_library as cls
from churn_library import *

logging.basicConfig(
    filename='logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

@pytest.fixture(scope="module")
def path():
    return r"datasets/bank_data.csv"

@pytest.fixture(scope="module")
def import_data_fixture(path):
    return import_data(path)


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

@pytest.fixture(scope="module")
def churn_column_fixture(import_data_fixture):
    return Churn_Column(import_data_fixture)

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
        assert os.path.isfile('images/eda/churn_histogram.png')
        assert os.path.isfile('images/eda/customer_age.png')
        assert os.path.isfile('images/eda/marital_status.png')
        assert os.path.isfile('images/eda/Total_Trans_Ct.png')
        assert os.path.isfile('images/eda/corr_heatmap.png')
    except AssertionError as err:
        logging.error("Error: Could not make some EDA plots")
        raise err
    finally:
        # Clean up: Remove the created image files after testing
        os.remove('images/eda/churn_histogram.png')
        os.remove('images/eda/customer_age.png')
        os.remove('images/eda/marital_status.png')
        os.remove('images/eda/Total_Trans_Ct.png')
        os.remove('images/eda/corr_heatmap.png')


def test_encoder_helper(churn_column_fixture, ):
	'''
	test encoder helper
	'''
	encoder_helper()

# def test_perform_feature_engineering(perform_feature_engineering):
	# '''
	# test perform_feature_engineering
	# '''


# def test_train_models(train_models):
	# '''
	# test train_models
	# '''


# if __name__ == "__main__":
#     # Run the test using pytest
#     # import_data(r"datasets/bank_data.csv")
# 	test_import(path)
# 	test_churn_column(import_data_fixture)
# 	test_eda(churn_column_fixture)

# if __name__ == "__main__":
#     # Run the test using pytest
#     pytest.main(["-v"])









