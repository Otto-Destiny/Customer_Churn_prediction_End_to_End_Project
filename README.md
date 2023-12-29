# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

### Project Overview

The Predict Customer Churn project aims to develop a robust and scalable solution for predicting customer churn in a business context. The project leverages a comprehensive data science model pipeline encapsulated in the churn_library.py script to enable efficient data import, exploratory data analysis (EDA), feature engineering, model training, and visualization of results. This README provides an in-depth overview of the project structure, files, data, and instructions for running the files.


### Project Structure
The project is organized as follows:

- `churn_library.py`: The core Python script containing modular functions for each step in the data science pipeline.

- `churn_library_logging_and_tests.py`: The test script containing test and logging functions for predicting customer churn.

- `logs/`: Directory to store log files.

- `datasets/bank_data.csv`: The dataset used for training and testing the predictive model.

- `models/`: Directory to store trained machine learning models.

- `images/`: Directory to save visualization outputs, including EDA plots, ROC curves, and feature importance plots.
## Files and data description
Overview of the files and data present in the root directory.

- **`churn_library.py`**
This script serves as the backbone of the project, encompassing the following functionalities:

  - **Data Import (`import_data`)**: Imports data from the specified CSV file path.

  - **Churn Column Creation (`churn_column`)**: Creates a 'Churn' column based on the 'Attrition_Flag' column.

  - **Exploratory Data Analysis (`perform_eda`)**: Generates visualizations of key features and correlations, saving them to the images/eda/ directory.

  - **Encoder Helper (`encoder_helper`)**: Encodes categorical features by creating new columns with the proportion of churn for each category.

  - **Feature Engineering (`perform_feature_engineering`)**: Splits the data into training and testing sets, readying them for model training.

  - **Generate Predictions (`generate_predictions`)**: Uses a trained model to generate predictions on the training and testing data.

  - **Classification Results Plotting (`plot_classification_results`)**: Plots and saves classification results for a given model, including classification reports.

  - **`classification_report_image:`** Function to produce a full classification report for training and testing results and store the report as an image for both models

  - **Feature Importance Plotting (`feature_importance_plot`)**: Creates and stores feature importances in the specified path.

  - **Model Training (`train_models`)**: Initializes and trains Random Forest and Logistic Regression models, performs hyperparameter tuning, and saves the best models to the models/ directory.

- **`churn_script_logging_and tests.py`**


   - **`test_import`**: Tests the data import function (`import_data`) and ensures the imported DataFrame has rows and columns.

   - **`test_churn_column`**: Tests the function (`churn_column`) that adds a 'Churn' column to the DataFrame.

   - **`test_eda`**: Tests the exploratory data analysis function (`perform_eda`) and checks if required image files are generated.

   - **`test_encoder_helper`**: Tests the encoder helper function (`encoder_helper`) and checks if encoding is successful.

   - **`test_perform_feature_engineering`**: Tests the feature engineering function (`perform_feature_engineering`) and checks various aspects such as dropped columns, train-test split, and test size.

   - **`test_train_models`**: Tests the model training function (`train_models`) and checks if models are fitted, ROC curve plots are generated, and models are saved.

   - **`test_generate_predictions`**: Tests the function (`generate_predictions`) that generates predictions and ensures the length of predicted labels matches the input data.

    - **`test_plot_classification_results`**: Tests the plot classification results function (`plot_classification_results`) and checks if classification results image is created.

    - **`test_classification_report_image`**: Tests the classification report image function (`classification_report_image`) and checks if classification report images are created.

    - **`test_feature_importance_plot`**: Tests the feature importance plot function (`feature_importance_plot`) and checks if the feature importance plot is created.


- **`datasets/bank_data.csv`**

The dataset, sourced from datasets/bank_data.csv, consists of both categorical and numerical columns. A column of interest is Attrition_Flag, from which the target column Churn is derived. Below is a breakdown of the columns into categorical and numerical groups:

**Categorical Columns**

1. `Gender`: The gender of the customer.
2. `Education_Level`: The educational level of the customer.
3. `Marital_Status`: The marital status of the customer.
4. `Income_Category`: The income category of the customer.
5. `Card_Category`: The category of the credit card held by the customer.
6. `Attrition_Flag`: The original column indicating customer status, used to create the Churn column.

**Numerical Columns**

1. `Customer_Age`: The age of the customer.
Dependent_count: The number of dependents of the customer.
2. `Months_on_book`: The number of months the customer has been on the books.
3. `Total_Relationship_Count`: The total number of products held by the customer.
4. `Months_Inactive_12_mon`: The number of months the customer has been inactive over the last twelve months.
5. `Contacts_Count_12_mon`: The number of contacts the customer has had over the last twelve months.
6. `Credit_Limit`: The credit limit of the customer.
7. `Total_Revolving_Bal`: The total revolving balance of the customer.
8. `Avg_Open_To_Buy`: The average amount available for credit purchases.
9. `Total_Amt_Chng_Q4_Q1`: The change in transaction amount from the fourth quarter to the first quarter.
10. `Total_Trans_Amt`: The total transaction amount.
11. `Total_Trans_Ct`: The total transaction count.
12. `Total_Ct_Chng_Q4_Q1`: The change in the total number of transactions from the fourth quarter to the first quarter.
13. `Avg_Utilization_Ratio`: The average card utilization ratio.


## Running Files
How do you run your files? What should happen when you run your files?
1. Ensure that all dependencies are installed. You can install them using the following command:

    ```bash
    pip install -r requirements_py3.6.txt
    ```
2. Clone the Repository: Clone the project repository to your local machine using the following command:
    ```bash
    git clone https://github.com/otto-destiny/predict-customer-churn.git
    ```
3. Run the Script: Navigate to the root directory of the project and Execute the main script `churn_library.py` to run the entire data science pipeline. Use the following command:
    ```bash
    python churn_library.py
    ```
4. Explore Results: The script should take about

Visualizations: Check the `images/` directory for visualizations generated during exploratory data analysis (EDA) and model evaluation.

Trained Models: Review the `models/` directory for saved machine learning models, including `rfc_model.pkl` (Random Forest Classifier) and `logistic_model.pkl` (Logistic Regression).

5. To run the tests:
   - **Data Import**: Set the path to your dataset in the `path()` fixture in `churn_library_logging_and_tests.py`.

        ```python
        @pytest.fixture(scope="module")
        def path():
            return r"./data/bank_data.csv"
        ```

   - **Run Tests**: Execute the tests using pytest with verbose output.

        ```bash
        pytest -v churn_library_logging_and_tests.py
        ```
6. The tes script should take about 3 - 5 mins to run.

*
**Author: Destiny Otto**


