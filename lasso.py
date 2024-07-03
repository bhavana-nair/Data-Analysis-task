import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import linregress,kurtosis, skew

merged_df = pd.read_csv('merged_data.csv')

X = merged_df.drop(['log_downloads', 'language', 'id', 'kld_values', 'downloads'], axis=1)
y = merged_df['log_downloads']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Setup LASSO regression
lasso = Lasso(random_state=42)
params = {'alpha': np.logspace(-3, 0, 100)} 

# hyperparameter tuning and cross-validation
lasso_cv = GridSearchCV(lasso, params, cv=5)
lasso_cv.fit(X_train_scaled, y_train)
best_alpha = lasso_cv.best_params_['alpha']
lasso_final = Lasso(alpha=best_alpha, random_state=42)
lasso_final.fit(X_train_scaled, y_train)

mse = mean_squared_error(y_test, lasso_final.predict(X_test_scaled))
r2 = r2_score(y_test, lasso_final.predict(X_test_scaled))

# Print results
print(f"Best alpha: {best_alpha}")
print(f"Mean Squared Error on test set: {mse}")
print(f"R-squared on test set: {r2}")

# Get feature importance (non-zero coefficients)
selected_features = X.columns[lasso_final.coef_ != 0]
print("Selected features:")
print(selected_features)
