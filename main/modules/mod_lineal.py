#MODULE lineal 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from modules.mod_init import *
from paths.paths import *
from columns.columns import *
from sklearn.model_selection import TimeSeriesSplit,cross_val_score

# READING file
df_preprocessing = pd.read_excel(path_preprocessing, header=None, skiprows=1, names=columns_preprocessing_order)

# SELECT X & y
X = df_preprocessing[['index_id']].values  # Asegúrate de seleccionar como una matriz 2D
y = df_preprocessing['close']

# SPLIT X & y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model with the training data
model.fit(X_train, y_train)

# Perform time series cross-validation
tscv = TimeSeriesSplit(n_splits=10)  
cv_scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=tscv)

# Print cross-validation scores
print("Cross-Validation Scores:")
for i, score in enumerate(cv_scores, 1):
    print(f'Fold {i}: {score}')

# Make predictions on the test data
y_pred = model.predict(X_test)  

# Print the predictions
#print('Predictions:')
#print(y_pred)

# Calculate the mean squared error (MSE)
mse = mean_squared_error(y_test, y_pred)
# Calculate R^2
r2 = r2_score(y_test, y_pred)

# Print the model parameters and evaluation metrics
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')