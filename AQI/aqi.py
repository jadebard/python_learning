import pandas as pd # data manipulation and analysis
import numpy as np # numerical computing
import matplotlib.pyplot as plt # plotting and visualization
import seaborn as sns # statistical data visualization

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("AQI Analysis and Prediction")
# Load the dataset
data = pd.read_csv('aqi_data.csv')

data = data.dropna()  # Drop rows with missing values
data.columns = [col.strip().lower() for col in data.columns]  # Strip whitespace from column names and convert to lowercase

# finding correlation between features in heatmap
# corr = data.corr()
# sns.heatmap(corr, annot=True, cmap='coolwarm')

# there is a better way for this
# excluding the actual value we are comparing against and excluding lat and long as they are not relevant for prediction
feature_exclude = {data.columns[0], data.columns[5], data.columns[6]}  

# Prepare the feature matrix X and target vector y by importing all columns except the target and irrelevant features
X = data[[col for col in data.columns if col not in feature_exclude]]

# The target variable is the first column (AQI)
y = data[data.columns[0]]

# Split the data into training and testing sets
# test_size=0.2 means 20% of the data will be used for testing, and 80% for training (i.e, 80% the model is learning from and comparing its results to the remaining 20% of the data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor 
# n_estimators is the number of trees in the forest, and random_state ensures reproducibility
# random_state is set to 42 for reproducibility, but you can choose any integer value
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Plotting actual vs predicted AQI values
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual AQI')
plt.plot(y_pred, label='Predicted AQI', alpha=0.7)
plt.title('Actual vs Predicted AQI')
plt.legend()
plt.show()