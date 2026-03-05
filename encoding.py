import pandas as pd
from sklearn.preprocessing import LabelEncoder
import ast

def label_encode(df, columns=None):
    le = LabelEncoder()
    if columns is None:
        columns = df.columns.tolist()
    elif isinstance(columns, str):
        columns = [columns]
    
    # Automatically encode any column that contains string representations of lists
    for col in columns:  # Update to use `columns` instead of `df.columns`
        if df[col].dtype == 'object':
            try:
                # Attempt to convert string representation of list to actual list
                df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) and isinstance(x, str) else x)
                
                # Flatten the list of items into a single series for label encoding
                flattened_items = [item for sublist in df[col].dropna() for item in (sublist if isinstance(sublist, list) else [sublist])]
                
                # Fit and transform the flattened items
                le.fit(flattened_items)
                
                # Create a mapping of original items to encoded values
                item_mapping = {item: encoded for item, encoded in zip(le.classes_, range(len(le.classes_)))}
                
                # Encode the original column
                df[col] = df[col].apply(lambda x: [item_mapping[item] for item in x] if isinstance(x, list) else item_mapping[x] if pd.notnull(x) else x)
            except (ValueError, SyntaxError):
                # If the column cannot be evaluated as lists, proceed with label encoding
                df[col] = le.fit_transform(df[col].astype(str))  # Convert to string for encoding

    return df


def one_hot_encode(df, columns=None):
    df = df.copy()
    if columns is None:
        columns = df.columns.tolist()
    elif isinstance(columns, str):
        columns = [columns]
    
    for col in df.columns:
        # Check if the column needs to be processed, based on column parameter
        if columns is None or col in columns:
            if df[col].dtype == 'object':
                try:
                    # Convert string representations of lists to actual lists
                    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                    
                    # Flatten lists by exploding the DataFrame and get dummies
                    df_exploded = df[col].explode()
                    one_hot = pd.get_dummies(df_exploded, prefix=col)
                    
                    # Sum across the index to handle duplicate indices created by explode
                    one_hot = one_hot.groupby(level=0).sum()
                    
                    # Drop original column and join one-hot encoded columns
                    df = df.drop(col, axis=1).join(one_hot)
                    
                except (ValueError, SyntaxError):
                    # Handle normal one-hot encoding for non-list data
                    one_hot = pd.get_dummies(df[col], prefix=col)
                    df = df.drop(col, axis=1).join(one_hot)
                    
    return df

