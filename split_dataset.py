import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df, test_size=0.3, target_column=None, shuffle=False):
    if target_column is None:
        raise ValueError("target_column must be specified.")
    
    if target_column not in df.columns:
        raise KeyError(f"'{target_column}' not found in dataframe columns.")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
    
    return X_train, X_test, y_train, y_test

def _save_data(X, y, filename, file_type):
    data = pd.concat([X, y], axis=1)
    
    if file_type == 'csv':
        data.to_csv(f'{filename}.csv', index=False)
    elif file_type == 'excel':
        data.to_excel(f'{filename}.xlsx', index=False)
    elif file_type == 'json':
        data.to_json(f'{filename}.json', orient='records')
    elif file_type == 'parquet':
        data.to_parquet(f'{filename}.parquet')
    elif file_type == 'txt':
        data.to_csv(f'{filename}.txt', index=False, sep='\t')
    elif file_type == 'html':
        data.to_html(f'{filename}.html', index=False)
    else:
        raise ValueError(f"Unsupported file format: {file_type}. Choose from 'csv', 'excel', 'json', 'parquet', 'txt', or 'html'.")
    
    print(f"Data saved as: {filename}.{file_type}")

def split_save_data(df, test_size=0.3, file_type='excel', target_column=None, shuffle=False):
    X_train, X_test, y_train, y_test = split_data(df, test_size, target_column, shuffle)
    
    _save_data(X_train, y_train, 'train_data', file_type)
    _save_data(X_test, y_test, 'test_data', file_type)
