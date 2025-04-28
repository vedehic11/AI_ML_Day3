# Housing Price Prediction

This project focuses on predicting housing prices using linear regression. The notebook `Day3.ipynb` walks through the process of importing a dataset, preprocessing the data, building a linear regression model, and evaluating its performance.

## Libraries Used

*   `pandas`: For data manipulation and analysis.
*   `numpy`: For numerical computations.
*   `matplotlib.pyplot`: For creating visualizations.
*   `seaborn`: For enhanced data visualization.
*   `sklearn.linear_model.LinearRegression`: For building the linear regression model.
*   `sklearn.model_selection.train_test_split`: For splitting the data into training and testing sets.
*   `sklearn.metrics`: For evaluating the model.
*   `kagglehub`: For downloading the dataset from Kaggle.

## Data Preprocessing

The notebook performs the following data preprocessing steps:

1.  **Importing the dataset:** The dataset is downloaded from Kaggle using the `kagglehub` library and loaded into a pandas DataFrame.
2.  **Handling missing values:** The notebook checks for missing values and describes how they would be handled (although the dataset used appears to have no missing values).
3.  **Outlier Removal:** Boxplots are used to visualize outliers in numerical columns, and outliers in the 'price' column are removed based on the IQR method.
4.  **Correlation Analysis:** A heatmap is generated to visualize the correlation between numerical features.

## Model Building and Evaluation

1.  **Splitting the data:** The dataset is split into training and testing sets using `train_test_split`.
2.  **Model training:** A linear regression model is trained on the training data using `LinearRegression.fit`.
3.  **Prediction:** The trained model is used to predict housing prices on the test data.
4.  **Evaluation:** The model is evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score.  A scatter plot of predicted vs. actual prices is also generated. The coefficients and intercept of the linear regression model are printed.

## Usage

To run this notebook, you need to have Python and the libraries listed above installed. You also need to have a Kaggle account and API key to download the dataset.

1.  Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
