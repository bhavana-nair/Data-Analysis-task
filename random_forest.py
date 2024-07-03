# Importing 
import pandas as pd
import numpy as np
from scipy.stats import linregress,kurtosis, skew
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

kld_scores_df = pd.read_csv('KLDscores.csv')
metadata_df = pd.read_csv('SPGC-metadata-2018-07-18.csv')
additional_features_df = pd.read_csv('extra_controls.csv')

kld_scores_df['kld_values'] = kld_scores_df['kld_values'].apply(lambda x: eval(x))

def compute_kld_statistics(kld_scores):
    kld_scores = np.array(kld_scores)
    avg_kld = np.mean(kld_scores)
    var_kld = np.var(kld_scores)
    std_kld = np.std(kld_scores)
    kurt_kld = kurtosis(kld_scores)
    skew_kld = skew(kld_scores)
    slope, intercept, r_value, p_value, std_err = linregress(range(len(kld_scores)), kld_scores)
    return avg_kld, var_kld, slope, std_kld, kurt_kld, skew_kld

kld_scores_df[['avg_kld', 'var_kld', 'slope_kld', 'std_kld', 'kurt_kld', 'skew_kld']] = kld_scores_df['kld_values'].apply(
    lambda scores: pd.Series(compute_kld_statistics(scores))
)

kld_scores_df.rename(columns={'filename': 'id'}, inplace=True)
merged_df = metadata_df.merge(kld_scores_df, on='id').merge(additional_features_df, on='id')

columns_to_drop = ['title', 'author', 'authoryearofdeath']
merged_df = merged_df.drop(columns=columns_to_drop, errors='ignore')

merged_df['log_downloads'] = np.log(merged_df['downloads'])
merged_df = merged_df.replace([np.inf, -np.inf], np.nan).dropna()

Y = merged_df['log_downloads']
X = merged_df[['avg_kld', 'var_kld', 'slope_kld', 'std_kld', 'authoryearofbirth',
               'speed', 'sentiment_avg', 'sentiment_vol', 'wordcount', 'subj2_war', 'subj2_adventure', 'subj2_comedy', 'subj2_biography', 'subj2_romance',
               'subj2_drama', 'subj2_fantasy', 'subj2_family', 'subj2_sciencefiction', 'subj2_action',
               'subj2_thriller', 'subj2_western', 'subj2_horror', 'subj2_mystery', 'subj2_crime',
               'subj2_history', 'subj2_periodicals', 'subj2_others']]

# X = merged_df.drop(['log_downloads', 'language', 'id', 'kld_values', 'downloads', 'subjects', 'type'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

rf_feature_importances = pd.Series(rf.feature_importances_, index=X.columns)

print("Random Forest feature importances:")
print(rf_feature_importances.sort_values(ascending=False))

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
rf_feature_importances.sort_values(ascending=False).plot(kind='bar')
plt.title('Feature Importances in the Random Forest Model')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.show()
