import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import linregress, kurtosis, skew
import statsmodels.api as sm
import matplotlib.pyplot as plt

merged_df = pd.read_csv('merged_data.csv')

genres = ['subj2_war', 'subj2_adventure', 'subj2_comedy', 'subj2_biography', 'subj2_romance',
          'subj2_drama', 'subj2_fantasy', 'subj2_family', 'subj2_sciencefiction', 'subj2_action',
          'subj2_thriller', 'subj2_western', 'subj2_horror', 'subj2_mystery', 'subj2_crime',
          'subj2_history', 'subj2_periodicals', 'subj2_others']

results = {}
r_squared_values = []

# Iterate over genres
for genre in genres:
    genre_df = merged_df[merged_df[genre] == 1]  # Subset data for the current genre
    
    if genre_df.shape[0] < 2:  # Check if there are at least 2 data points
            r_squared_values.append(0)
            continue
     
    X = genre_df[['avg_kld', 'var_kld', 'slope_kld', 'std_kld', 'authoryearofbirth',
               'speed', 'sentiment_avg', 'sentiment_vol', 'wordcount', 'subj2_sciencefiction', 'subj2_comedy','subj2_war', 'subj2_adventure', 'subj2_thriller', 
                'subj2_periodicals', 'subj2_fantasy', 'subj2_horror', 'subj2_mystery', 'subj2_others', 'subj2_family', 'subj2_romance']]
    
    y = genre_df['log_downloads']
    
    # Add constant to X
    X = sm.add_constant(X)
    
    # Fit OLS model
    model = sm.OLS(y, X)
    results[genre] = model.fit()
    r_squared_values.append(results[genre].rsquared)

for genre, result in results.items():
    print(f"Genre: {genre}")
    print(result.summary())
    print("\n")

plt.figure(figsize=(10, 6))
plt.bar(genres, r_squared_values, color='blue')
plt.xlabel('Genres')
plt.ylabel('R-squared')
plt.title('R-squared Values for Different Genres')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()