import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

 
np.random.seed(0)
X = np.random.rand(100, 2)   
y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.rand(100)   

 
data = pd.DataFrame({'X1': X[:, 0], 'X2': X[:, 1], 'y': y})

 
X_train, X_test, y_train, y_test = train_test_split(data[['X1', 'X2']], data['y'], test_size=0.2, random_state=42)

 
model = LinearRegression()

 
model.fit(X_train, y_train)

 
y_pred = model.predict(X_test)

 
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
 
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
