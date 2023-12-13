import os
import logging
import pytest
# import churn_library as cls
from churn_library import import_data, Churn_Column

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# @pytest.fixture(scope="module")
def test_import(import_data):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = import_data(r"datasets/bank_data.csv")
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

# @pytest.fixture(scope="module")
def test_churn_column(Churn_Column):
	try:
		df = Churn_Column(import_data())
		logging.info("SUCCESS: Churn Column added successfully")
		assert 'Churn' in list(df.columns)
	except AssertionError as err:
		logging.error("Failed to add column!")
		raise err
	except KeyError as err:
		logging.error("Attrition_Flag column was not found!")
		raise err

		


# def test_eda(perform_eda):
	#'''
	# test perform eda function
	# '''
	# Ensure that the 'images/eda' directory exists
    # os.makedirs('images/eda', exist_ok=True)

    # try:
	# 	perform_eda()
    #     # Check if the required image files are generated
    #     assert os.path.isfile('images/eda/churn_histogram.png')
    #     assert os.path.isfile('images/eda/customer_age.png')
    #     assert os.path.isfile('images/eda/marital_status.png')
    #     assert os.path.isfile('images/eda/Total_Trans_Ct.png')
    #     assert os.path.isfile('images/eda/corr_heatmap.png')
    # except Exception as err:
    #     raise err
    # finally:
    #     # Clean up: Remove the created image files after testing
    #     os.remove('images/eda/churn_histogram.png')
    #     os.remove('images/eda/customer_age.png')
    #     os.remove('images/eda/marital_status.png')
    #     os.remove('images/eda/Total_Trans_Ct.png')
    #     os.remove('images/eda/corr_heatmap.png')


# def test_encoder_helper(encoder_helper):
	# '''
	# test encoder helper
	# '''


# def test_perform_feature_engineering(perform_feature_engineering):
	# '''
	# test perform_feature_engineering
	# '''


# def test_train_models(train_models):
	# '''
	# test train_models
	# '''


if __name__ == "__main__":
    # Run the test using pytest
    pytest.main([__file__, '-v'])








