import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from .loader import *

def check_outliers(df, columns=None):
    """Detects outliers in the DataFrame using the IQR method and calculates the percentage of outliers."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns

    outliers_info = {}
    for column in columns:
        if column not in df.columns:
            print(f"Column '{column}' does not exist in the DataFrame.")
            continue

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        outlier_indices = df.index[outlier_mask]

        num_outliers = len(outlier_indices)
        total_data_points = len(df[column])
        outlier_percentage = (num_outliers / total_data_points) * 100

        outliers_info[column] = {
            'count': num_outliers,
            'indices': outlier_indices.tolist(),
            'percentage': outlier_percentage
        }

    return outliers_info

def _check(df, columns=None):
    """Detects outliers in the DataFrame using the IQR method and calculates the percentage of outliers."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns

    outliers_info = {}
    for column in columns:
        if column not in df.columns:
            print(f"Column '{column}' does not exist in the DataFrame.")
            continue

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        outlier_indices = df.index[outlier_mask]

        num_outliers = len(outlier_indices)
        total_data_points = len(df[column])
        outlier_percentage = (num_outliers / total_data_points) * 100

        outliers_info[column] = {
            'count': num_outliers,
            'indices': outlier_indices.tolist(),
            'percentage': outlier_percentage
        }

    return outliers_info

def plot_outliers(df):
    """Visualizes outliers in the DataFrame."""
    outliers_dict = _check(df)  # Detect outliers
    plt.figure(figsize=(12, 6))
    for i, column in enumerate(outliers_dict.keys(), start=1):
        plt.subplot(2, 3, i)
        sns.boxplot(data=df[column])
        plt.title(f'Outliers in {column}')

        # Highlight the outliers
        outlier_indices = outliers_dict[column]['indices']
        outlier_values = df.loc[outlier_indices, column]
        plt.scatter(outlier_indices, outlier_values, color='red', label='Outliers', zorder=5)

    plt.tight_layout()
    plt.legend()
    plt.show()




def fill_outliers_with_mean(df, columns=None):
    """Replace outliers in the DataFrame with the mean value, excluding nulls and outliers in the mean calculation."""
    if df.empty:
        print("The DataFrame is empty.")
        return df

    # If columns is a string, convert it to a list
    if isinstance(columns, str):
        columns = [columns]
    
    df_copy = df.copy()
    
    if columns is None:
        # Select only numeric columns if no specific columns are provided
        columns = df_copy.select_dtypes(include=['number']).columns
    
    for column in columns:
        if column in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[column]):
            Q1 = df_copy[column].quantile(0.25)
            Q3 = df_copy[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Calculate the mean excluding nulls and outliers
            mean_value = df_copy[(df_copy[column] >= lower_bound) & (df_copy[column] <= upper_bound)][column].mean()
            
            # Replace outliers with the calculated mean
            df_copy[column] = np.where((df_copy[column] < lower_bound) | (df_copy[column] > upper_bound), mean_value, df_copy[column])
    
    return df_copy

def fill_outliers_with_median(df, columns=None):
    """Replace outliers in the DataFrame with the median value, excluding nulls and outliers in the median calculation."""
    if df.empty:
        print("The DataFrame is empty.")
        return df

    # If columns is a string, convert it to a list
    if isinstance(columns, str):
        columns = [columns]
    
    df_copy = df.copy()
    
    if columns is None:
        # Select only numeric columns if no specific columns are provided
        columns = df_copy.select_dtypes(include=['number']).columns
    
    for column in columns:
        if column in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[column]):
            Q1 = df_copy[column].quantile(0.25)
            Q3 = df_copy[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Calculate the median excluding nulls and outliers
            median_value = df_copy[(df_copy[column] >= lower_bound) & (df_copy[column] <= upper_bound)][column].median()
            
            # Replace outliers with the calculated median
            df_copy[column] = np.where((df_copy[column] < lower_bound) | (df_copy[column] > upper_bound), median_value, df_copy[column])
    
    return df_copy

def fill_outliers_with_mode(df, columns=None):
    """Replace outliers in the DataFrame with the mode value, excluding nulls and outliers in the mode calculation."""
    if df.empty:
        print("The DataFrame is empty.")
        return df

    # If columns is a string, convert it to a list
    if isinstance(columns, str):
        columns = [columns]
    
    df_copy = df.copy()
    
    if columns is None:
        # Select only numeric columns if no specific columns are provided
        columns = df_copy.select_dtypes(include=['number']).columns
    
    for column in columns:
        if column in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[column]):
            Q1 = df_copy[column].quantile(0.25)
            Q3 = df_copy[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Calculate the mode excluding nulls and outliers
            mode_value = df_copy[(df_copy[column] >= lower_bound) & (df_copy[column] <= upper_bound)][column].mode()
            mode_value = mode_value[0] if not mode_value.empty else np.nan  # Get the first mode or NaN if empty
            
            # Replace outliers with the calculated mode
            df_copy[column] = np.where((df_copy[column] < lower_bound) | (df_copy[column] > upper_bound), mode_value, df_copy[column])
    
    return df_copy

def fill_outliers_with_random_values(df, columns=None):
    """Replace outliers in the DataFrame with random values from the same column, excluding nulls and outliers in the random selection."""
    if df.empty:
        print("The DataFrame is empty.")
        return df

    # If columns is a string, convert it to a list
    if isinstance(columns, str):
        columns = [columns]
    
    df_copy = df.copy()
    
    if columns is None:
        # Select only numeric columns if no specific columns are provided
        columns = df_copy.select_dtypes(include=['number']).columns
    
    for column in columns:
        if column in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[column]):
            Q1 = df_copy[column].quantile(0.25)
            Q3 = df_copy[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify non-outlier values
            non_outliers = df_copy[(df_copy[column] >= lower_bound) & (df_copy[column] <= upper_bound)][column]
            
            # Replace outliers with random values from non-outliers
            df_copy[column] = np.where((df_copy[column] < lower_bound) | (df_copy[column] > upper_bound),
                                         np.random.choice(non_outliers, size=df_copy[column].shape[0]),
                                         df_copy[column])
    
    return df_copy

def fill_outliers_with_values(df, fill_values):
    """Replace outliers in the DataFrame with specified values from a dictionary for all numeric columns."""
    if df.empty:
        print("The DataFrame is empty.")
        return df

    df_copy = df.copy()
    
    # Select only numeric columns
    numeric_columns = df_copy.select_dtypes(include=['number']).columns
    
    for column in numeric_columns:
        if column in fill_values:
            Q1 = df_copy[column].quantile(0.25)
            Q3 = df_copy[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Replace outliers with specified values from the dictionary
            fill_value = fill_values[column]
            df_copy[column] = np.where((df_copy[column] < lower_bound) | (df_copy[column] > upper_bound), fill_value, df_copy[column])
    
    return df_copy



def remove_rows_with_outliers(df, count=1):
    """Keep only rows with fewer than the specified number of outliers based on the Interquartile Range (IQR) method, considering only numeric columns."""

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    # Calculate Q1 (25th percentile) and Q3 (75th percentile) for numeric columns
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    
    # Calculate IQR
    IQR = Q3 - Q1
    
    # Determine the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Create a boolean mask for outliers in numeric columns
    outlier_mask = (numeric_df < lower_bound) | (numeric_df > upper_bound)
    
    # Count the number of outliers in each row based on numeric columns
    outlier_counts = outlier_mask.sum(axis=1)
    
    # Filter the original DataFrame to keep only rows with fewer than the specified number of outliers
    df_kept = df[outlier_counts < count]
    
    return df_kept



