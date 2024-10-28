# Modx
GDP Prediction Model
This project develops a Deep Neural Network (DNN) model to predict a target economic indicator (productivity_index) based on multiple input features. The model leverages various machine learning and deep learning techniques to achieve accurate predictions and evaluate model performance.

# Project Structure
Data: The main data file, main.csv, contains the input features and target variable for model training and evaluation.
Notebook: The file testing.ipynb contains an interactive Jupyter Notebook for model building, training, evaluation, and experimentation.
Files
main.csv - Contains the raw data, with features such as GDP_growth, R&D_expenditure, patent_activity, and employment_trends and the target variable productivity_index.
testing.ipynb - The Jupyter Notebook used to run the model pipeline and visualize results.
Requirements
Ensure you have the following libraries installed:

bash
Copy code
pip install pandas numpy scikit-learn tensorflow matplotlib
# Model Pipeline
1. Data Preprocessing
Feature Selection: Select key economic features from the dataset.
Feature Scaling: Apply standardization to improve model performance.
Train-Test Split: Divide data into training and testing sets (80% training, 20% testing).
2. Model Architecture
A Sequential Neural Network is implemented with:

Dense Layers: The model has two hidden layers:
64 nodes with ReLU activation
32 nodes with ReLU activation
Output Layer: One node for regression output.
3. Model Training and Evaluation
Training: The model trains over 1000 epochs with a batch size of 16, using a validation split to monitor performance.
Loss Function: Mean Squared Error (MSE).
Metric: Mean Absolute Error (MAE) and R-squared score to evaluate model accuracy.
4. Results Visualization
A plot of training and validation loss over epochs provides insight into the model’s learning progression.

# Usage
To run the model, open testing.ipynb and execute the cells sequentially. The notebook will load the dataset, preprocess data, build and train the model, and finally evaluate the results.

# Key Metrics
Test Loss (MSE): Measures the average squared difference between predicted and actual values.
Mean Absolute Error (MAE): Reflects the average magnitude of errors in predictions.
R-squared Score: Provides an “accuracy” measure by indicating the proportion of variance explained by the model.
# Sample Output
Upon running the code, expected outputs include:

Model loss and MAE values on test data.
R-squared score (in percentage) indicating model performance.
A training history plot showing loss reduction over epochs.
