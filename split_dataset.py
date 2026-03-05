import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df, test_size=0.3, target_column=None, shuffle=False):
    # Check if target_column is provided
    if target_column is None:
        raise ValueError("target_column must be specified.")
    
    # Ensure the target column exists in the dataframe
    if target_column not in df.columns:
        raise KeyError(f"'{target_column}' not found in dataframe columns.")
    
    # Split the data into features (X) and target (y)
    X = df.drop(columns=[target_column])  # Features (all columns except the target column)
    y = df[target_column]  # Target column
    
    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
    
    return X_train, X_test, y_train, y_test

def _save_data(X, y, filename, file_type):
    # Concatenate X (features) and y (target) into a single dataframe for saving
    data = pd.concat([X, y], axis=1)
    
    # Save the data in the specified format
    if file_type == 'csv':
        data.to_csv(f'{filename}.csv', index=False)
    elif file_type == 'excel':
        data.to_excel(f'{filename}.xlsx', index=False)
    elif file_type == 'json':
        data.to_json(f'{filename}.json', orient='records')
    elif file_type == 'parquet':
        data.to_parquet(f'{filename}.parquet')
    elif file_type == 'txt':
        data.to_csv(f'{filename}.txt', index=False, sep='\t')  # Save as tab-delimited text file
    elif file_type == 'html':
        data.to_html(f'{filename}.html', index=False)
    else:
        raise ValueError(f"Unsupported file format: {file_type}. Choose from 'csv', 'excel', 'json', 'parquet', 'txt', or 'html'.")
    
    print(f"Data saved as: {filename}.{file_type}")

def split_save_data(df, test_size=0.3, file_type='excel', target_column=None, shuffle=False):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = split_data(df, test_size, target_column, shuffle)
    
    # Save the train and test datasets using the internal _save_data function
    _save_data(X_train, y_train, 'train_data', file_type)
    _save_data(X_test, y_test, 'test_data', file_type)
