import pandas as pd
import numpy as np
from scipy.stats import linregress,kurtosis, skew
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load the KLD scores file
kld_scores_df = pd.read_csv('KLDscores.csv')
metadata_df = pd.read_csv('SPGC-metadata-2018-07-18.csv')
additional_features_df = pd.read_csv('extra_controls.csv')

# Convert the kld_scores column from string representation of list to actual list
kld_scores_df['kld_values'] = kld_scores_df['kld_values'].apply(lambda x: eval(x))

# Function to compute KLD statistics
def compute_kld_statistics(kld_scores):
    kld_scores = np.array(kld_scores)
    avg_kld = np.mean(kld_scores)
    var_kld = np.var(kld_scores)
    std_kld = np.std(kld_scores)
    kurt_kld = kurtosis(kld_scores)
    skew_kld = skew(kld_scores)
    slope, intercept, r_value, p_value, std_err = linregress(range(len(kld_scores)), kld_scores)
    return avg_kld, var_kld, slope, std_kld, kurt_kld, skew_kld

# Apply the function to each row and create new columns for the statistics
kld_scores_df[['avg_kld', 'var_kld', 'slope_kld', 'std_kld', 'kurt_kld', 'skew_kld']] = kld_scores_df['kld_values'].apply(
    lambda scores: pd.Series(compute_kld_statistics(scores))
)

# Display the resulting DataFrame
print(kld_scores_df.head())

kld_scores_df.rename(columns={'filename': 'id'}, inplace=True)

merged_df = metadata_df.merge(kld_scores_df, on='id').merge(additional_features_df, on='id')
columns_to_drop = ['title', 'author', 'authoryearofdeath', 'subjects', 'type']
merged_df = merged_df.drop(columns=columns_to_drop, errors='ignore')
merged_df['log_downloads'] = np.log(merged_df['downloads'])

# Handle missing or infinite values (drop rows with NaNs or infs)
merged_df = merged_df.replace([np.inf, -np.inf], np.nan).dropna()

merged_df.to_csv('merged_data.csv', index=False)