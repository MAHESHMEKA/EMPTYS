import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from .loader import *


# Existing Functions

def plot_missing_values(df):
    missing_count = df.isnull().sum()
    total_rows = df.shape[0]
    missing_percentage = (missing_count / total_rows) * 100
    total_missing_count = missing_count.sum()
    total_missing_percentage = (total_missing_count / (total_rows * df.shape[1])) * 100

    plt.figure(figsize=(12, 6))
    bar_plot = sns.barplot(x=missing_count.index, y=missing_count.values, palette='viridis', hue=missing_count.index, legend=False)
    plt.title('Count of Missing Values per Column')
    plt.ylabel('Number of Missing Values')
    plt.xlabel('Columns')
    plt.xticks(rotation=45)

    for p in bar_plot.patches:
        count = int(p.get_height())
        index = int(p.get_x() + p.get_width() / 2)
        percentage = missing_percentage.iloc[index].round(2)
        bar_plot.annotate(f'{count} ({percentage}%)', 
                          (p.get_x() + p.get_width() / 2., count),
                          ha='center', va='bottom', color='black', fontsize=10)

    plt.figure(figsize=(12, 6))
    plt.bar('Total Missing', total_missing_count, color='red', alpha=0.6)
    plt.title('Total Count of Missing Values')
    plt.ylabel('Number of Missing Values')
    plt.annotate(f'{total_missing_count} ({total_missing_percentage:.2f}%)', 
                 ('Total Missing', total_missing_count),
                 ha='center', va='bottom', color='black', fontsize=10)

    plt.tight_layout()
    plt.show()

def check_missing_values(df):
    total_rows = df.shape[0]
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / total_rows) * 100

    missing_data = pd.DataFrame({
        'Missing Count': missing_values,
        'Missing Percentage': missing_percentage.round(2)
    })
    
    missing_data = missing_data[missing_data['Missing Count'] > 0]

    total_missing = missing_values.sum()
    total_missing_percentage = (total_missing / (total_rows * df.shape[1])) * 100
    print(f"\nTotal Missing Values: {total_missing} ({total_missing_percentage:.2f}%)")

    return missing_data



# Extended Functions

def remove_columns_with_missing(df, threshold=0.3):
    """Remove columns with a missing value ratio above the specified threshold and save history if required."""
    missing_ratios = df.isnull().mean()
    columns_to_drop = missing_ratios[missing_ratios >= threshold].index.tolist()

    # Remove the columns from the DataFrame
    df_dropped = df.drop(columns=columns_to_drop)

    return df_dropped


def remove_rows_with_missing(df, count=1):
    """Keep only rows with fewer than the specified number of missing values and save history if required."""

    # Filter to keep rows with missing values less than the specified count
    df_kept = df[df.isnull().sum(axis=1) < count]
    
    return df_kept


def fill_missing_with_mean(df, columns=None):
    # Ensure `columns` is a list if a single column name is passed
    if isinstance(columns, str):
        columns = [columns]
    
    # If `columns` is None, default to all columns in the DataFrame
    if columns is None:
        columns = df.columns
    
    # Create a copy of the DataFrame to avoid chaining warnings and handle inplace issues
    df_copy = df.copy()
    
    for column in columns:
        if column in df_copy.columns:
            # Check if the column is numeric before attempting to calculate the mean
            if pd.api.types.is_numeric_dtype(df_copy[column]):
                df_copy[column] = df_copy[column].fillna(df_copy[column].mean())
            else:
                print(f"Column '{column}' is not numeric and was skipped for mean imputation.")
        else:
            print(f"Column '{column}' does not exist in the DataFrame.")
    
    return df_copy



def fill_missing_with_median(df, columns=None):
    # Ensure `columns` is a list if a single column name is passed
    if isinstance(columns, str):
        columns = [columns]
    
    # If `columns` is None, use all columns in the DataFrame
    if columns is None:
        columns = df.columns
    
    # Create a copy of the DataFrame to avoid inplace chaining warnings
    df_copy = df.copy()
    
    for column in columns:
        if column in df_copy.columns:
            # Check if the column is numeric before calculating the median
            if pd.api.types.is_numeric_dtype(df_copy[column]):
                df_copy[column] = df_copy[column].fillna(df_copy[column].median())
            else:
                print(f"Column '{column}' is not numeric and was skipped for median imputation.")
        else:
            print(f"Column '{column}' does not exist in the DataFrame.")
    
    return df_copy


def fill_missing_with_mode(df, columns=None):
    # Ensure `columns` is a list if a single column name is passed
    if isinstance(columns, str):
        columns = [columns]
    
    # If `columns` is None, use all columns in the DataFrame
    if columns is None:
        columns = df.columns
    
    # Create a copy of the DataFrame to avoid inplace chaining warnings
    df_copy = df.copy()
    
    for column in columns:
        if column in df_copy.columns:
            # Check if the column is numeric or categorical/ordinal (mode can be applied to both)
            if pd.api.types.is_numeric_dtype(df_copy[column]) or pd.api.types.is_categorical_dtype(df_copy[column]) or pd.api.types.is_object_dtype(df_copy[column]):
                # Fill with the most frequent value (mode)
                df_copy[column] = df_copy[column].fillna(df_copy[column].mode()[0])
            else:
                print(f"Column '{column}' is not suitable for mode imputation and was skipped.")
        else:
            print(f"Column '{column}' does not exist in the DataFrame.")
    
    return df_copy


def fill_missing_with_in_order(df, columns=None):
    if columns is None:
        columns = df.columns
    if isinstance(columns, str):
        columns = [columns]
    
    for column in columns:
        if column not in df.columns:
            print(f"Column '{column}' does not exist in the DataFrame.")
            continue
        
        # Ensure the column is numeric
        df[column] = pd.to_numeric(df[column], errors='coerce')
        
        # Iterate through the DataFrame
        for i in range(len(df[column])):
            if pd.isna(df[column].iloc[i]):
                # Fill with the previous value + 1
                if i > 0 and pd.notna(df[column].iloc[i - 1]):
                    df.loc[i, column] = df[column].iloc[i - 1] + 1
                else:
                    # If there's no valid previous value, you can decide how to handle it
                    # For example, you might want to start from 1 or leave it as NaN
                    df.loc[i, column] = 1  # or you could choose to leave it as NaN

    return df


def fill_missing_with_random_values(df, columns=None):
    if isinstance(columns, str):
        columns = [columns]
    if columns is None:
        columns = df.columns
    for column in columns:
        if column not in df.columns:
            print(f"Column '{column}' does not exist in the DataFrame.")
            continue
        
        non_null_values = df[column].dropna().values
        if len(non_null_values) == 0:
            print(f"No non-null values available to sample for column '{column}'.")
            continue
        
        random_samples = np.random.choice(non_null_values, size=df[column].isnull().sum(), replace=True)
        df.loc[df[column].isnull(), column] = random_samples
    return df

def fill_missing_with_values(df, fill_dict):
    for column, value in fill_dict.items():
        if column not in df.columns:
            print(f"Column '{column}' does not exist in the DataFrame.")
            continue
        
        # Fill NaN values in the specified column with the provided value
        df[column] = df[column].where(pd.notna(df[column]), value)
    
    return df






