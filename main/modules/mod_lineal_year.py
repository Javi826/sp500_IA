#MODULE lineal year

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from modules.mod_init import *
from paths.paths import *
from columns.columns import *

# READ file
df_preprocessing = pd.read_excel(path_preprocessing, header=None, skiprows=1, names=columns_preprocessing_order)

# 
df_preprocessing['date'] = pd.to_datetime(df_preprocessing['date'])

# 
unique_years = df_preprocessing['date'].dt.year.unique()

# BUCLE for years
for year in unique_years:
    # FILTER years
    df_year = df_preprocessing[df_preprocessing['date'].dt.year == year]
    
    # Feature (X) and target variable (y) selection
    X = df_year[['index_id']].values  # Make sure to select as a 2D array
    y = df_year['close']
    
    # Splitting into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a linear regression model
    model = LinearRegression()
    
    # Train the model with the training data
    model.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    # Calculate the mean squared error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    
    # Calculate R^2
    r2 = r2_score(y_test, y_pred)
    
    # Print results for the current year
    print(f'Year: {year}')
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    print(f'Coefficients: {model.coef_}')
    print(f'Intercept: {model.intercept_}')
    print('---------------------')


# Plotting the regression line and data points
plt.scatter(X_test, y_test, color='black', label='Test Data')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Regression Line')
plt.title('Linear Regression Fit')
plt.xlabel('Index ID')
plt.ylabel('Close Price')
plt.legend()
plt.show()



