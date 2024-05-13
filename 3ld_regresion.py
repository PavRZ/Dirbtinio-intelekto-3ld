import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


import matplotlib.pyplot as plt

def plot_scatter_diagrams(linear_pred, poly_pred, rf_pred, rnn_pred, y_test):
    plt.figure(figsize=(16, 10))

    # Linear Regression
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, linear_pred, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
    plt.title('Linear Regression: Actual vs. Predicted')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.grid(True)

    # Polynomial Regression
    plt.subplot(2, 2, 2)
    plt.scatter(y_test, poly_pred, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
    plt.title('Polynomial Regression: Actual vs. Predicted')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.grid(True)

    # Random Forest Regression
    plt.subplot(2, 2, 3)
    plt.scatter(y_test, rf_pred, color='red')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
    plt.title('Random Forest Regression: Actual vs. Predicted')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.grid(True)

    # RNN
    plt.subplot(2, 2, 4)
    plt.scatter(y_test, rnn_pred, color='purple')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
    plt.title('RNN: Actual vs. Predicted')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.grid(True)

    plt.tight_layout()

    plt.savefig("original vs augmented images.png")
    plt.show()

def write_model_results_single_example(model_name, correct_value, predicted_value):
    print("Model name:", model_name)
    print("Correct value:", correct_value.iloc[4], ", Predicted value:", predicted_value[4])

"""
## 1st step importing data
"""
df = pd.read_csv("data/house_price.csv")

"""
## 2nd step data processing
"""
# null data removal
df.dropna(inplace=True)
# Removing outliers --- VERY IMPORTANT, THE SCATTER DIAGRAMS WERE WRONG BEFORE REMOVING OUTLIERS
# Define a function to remove outliers based on IQR
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered

# Specify the column(s) you want to remove outliers from
columns_to_filter = ['price', 'sqft_living']  # Adjust this list as needed

# Remove outliers from each specified column
for column in columns_to_filter:
    df = remove_outliers_iqr(df, column)

# unnecessary data removal
df = df.drop(columns=['date','sqft_lot', 'waterfront', 'view',  'condition', 'sqft_above', 'sqft_basement', 'yr_renovated', 'street', 'city', 'statezip','country'])

print(df.head())

# Encoding categorical variables
# If there are categorical columns, encode them using LabelEncoder
label_encoders = {}
categorical_columns = ["bedrooms", "bathrooms", "floors"]  # Define potential categorical columns
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Scaling numerical features
scaler = StandardScaler()
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

print(df.head())

"""
## Regresion moddeling
"""
# Splitting data into training and testing sets
X = df.drop(columns=['price'])
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3rd step: Implementing regression algorithms
# Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Polynomial Regression
poly_reg = PolynomialFeatures(degree=2)
X_poly_train = poly_reg.fit_transform(X_train)
X_poly_test = poly_reg.transform(X_test)
poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)

# Random Forest Regression
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

# 4th step: Evaluating models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    return mae, mse

linear_mae, linear_mse = evaluate_model(linear_reg, X_test, y_test)
poly_mae, poly_mse = evaluate_model(poly_model, X_poly_test, y_test)
rf_mae, rf_mse = evaluate_model(rf_reg, X_test, y_test)

print("Linear Regression:")
print("MAE:", linear_mae)
print("MSE:", linear_mse)
print("\nPolynomial Regression:")
print("MAE:", poly_mae)
print("MSE:", poly_mse)
print("\nRandom Forest Regression:")
print("MAE:", rf_mae)
print("MSE:", rf_mse)

# RNN
# Reshape data for RNN
X_train_rnn = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
X_test_rnn = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))

rnn_model = Sequential([
    LSTM(50, activation='relu', input_shape=(1, X_train.shape[1])),
    Dense(1)
])
rnn_model.compile(optimizer='adam', loss='mse')
rnn_model.fit(X_train_rnn, y_train, epochs=15, batch_size=32, verbose=0)

rnn_mae, rnn_mse = evaluate_model(rnn_model, X_test_rnn, y_test)
print("\nRNN:")
print("MAE:", rnn_mae)
print("MSE:", rnn_mse)

# Linear Regression predictions
linear_pred = linear_reg.predict(X_test)

# Polynomial Regression predictions
poly_pred = poly_model.predict(X_poly_test)

# Random Forest Regression predictions
rf_pred = rf_reg.predict(X_test)

# RNN model predictions
rnn_pred = rnn_model.predict(X_test_rnn)

# Plotting scatter diagrams of each model predictions
plot_scatter_diagrams(linear_pred, poly_pred, rf_pred, rnn_pred, y_test)

# Writing original price and predicted price of each model
write_model_results_single_example("Linear Regression", y_test, linear_pred)
write_model_results_single_example("Polynomial Regression", y_test, poly_pred)
write_model_results_single_example("Random Forest Regression", y_test, rf_pred)
write_model_results_single_example("RNN", y_test, rnn_pred)