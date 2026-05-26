import pandas as pd
from sklearn.preprocessing import LabelEncoder
import ast

def label_encode(df, columns=None):
    le = LabelEncoder()
    if columns is None:
        columns = df.columns.tolist()
    elif isinstance(columns, str):
        columns = [columns]
    
    for col in columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) and isinstance(x, str) else x)
                
                flattened_items = [item for sublist in df[col].dropna() for item in (sublist if isinstance(sublist, list) else [sublist])]
                
                le.fit(flattened_items)
                
                item_mapping = {item: encoded for item, encoded in zip(le.classes_, range(len(le.classes_)))}
                
                df[col] = df[col].apply(lambda x: [item_mapping[item] for item in x] if isinstance(x, list) else item_mapping[x] if pd.notnull(x) else x)
            except (ValueError, SyntaxError):
                df[col] = le.fit_transform(df[col].astype(str))

    return df


def one_hot_encode(df, columns=None):
    df = df.copy()
    if columns is None:
        columns = df.columns.tolist()
    elif isinstance(columns, str):
        columns = [columns]
    
    for col in df.columns:
        if columns is None or col in columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                    
                    df_exploded = df[col].explode()
                    one_hot = pd.get_dummies(df_exploded, prefix=col)
                    
                    one_hot = one_hot.groupby(level=0).sum()
                    
                    df = df.drop(col, axis=1).join(one_hot)
                    
                except (ValueError, SyntaxError):
                    one_hot = pd.get_dummies(df[col], prefix=col)
                    df = df.drop(col, axis=1).join(one_hot)
                    
    return df
