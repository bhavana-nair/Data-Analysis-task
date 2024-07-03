import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

merged_df = pd.read_csv('merged_data.csv')

Y = merged_df['log_downloads']

# KLD measures + controls used 
X = merged_df[['avg_kld', 'var_kld', 'slope_kld', 'std_kld', 'skew_kld', 'authoryearofbirth',
               'speed', 'sentiment_avg', 'sentiment_vol', 'wordcount', 
               'subj2_sciencefiction', 'subj2_comedy','subj2_war', 'subj2_adventure', 'subj2_thriller', 
                 'subj2_periodicals', 'subj2_fantasy', 'subj2_horror', 'subj2_mystery', 'subj2_others',
                 'subj2_family', 'subj2_romance']]

X_with_const = sm.add_constant(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_with_const, Y, test_size=0.2, random_state=42)

# Fit OLS model on training data
ols_model = sm.OLS(Y_train, X_train).fit()

# Generate predictions on test set
predictions = ols_model.predict(X_test)

# Calculate residuals and MSE on test set
residuals = Y_test - predictions
mse = mean_squared_error(Y_test, predictions)

print(f"Mean Squared Error on test set: {mse}")
print(ols_model.summary())

# Extract the coefficients
ols_coefficients = pd.Series(ols_model.params, index=X_with_const.columns)
ols_coefficients_abs = ols_coefficients.abs().sort_values(ascending=False)

print("Sorted OLS Model Coefficients by Absolute Value:")
print(ols_coefficients_abs)

# Print original coefficients
print("\nOriginal OLS Model Coefficients:")
print(ols_coefficients)

# Plot the coefficients
plt.figure(figsize=(12, 8))

# Plot absolute values
plt.subplot(1, 2, 1)
ols_coefficients_abs.sort_values().plot(kind='barh', color='skyblue')
plt.title('Absolute Coefficient Values')
plt.xlabel('Absolute Coefficient Value')
plt.ylabel('Features')

# Plot original signed values
plt.subplot(1, 2, 2)
ols_coefficients.loc[ols_coefficients_abs.index].sort_values().plot(kind='barh', color='lightgreen')
plt.title('Original Coefficient Values')
plt.xlabel('Coefficient Value')
plt.ylabel('Features')

plt.tight_layout()
plt.show()