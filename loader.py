import pandas as pd
import copy
import os
import datetime

def load(data_source):
    # Check if data_source is a DataFrame and return it directly if so
    if isinstance(data_source, pd.DataFrame):
        return data_source

    # Check if data_source is a string (path) or a dictionary/list (data)
    if isinstance(data_source, str):
        # Get the directory path of the current file (inside the package)
        package_dir = os.path.dirname(__file__)
        
        # Construct the full path if the file is located within the package or external path
        full_path = os.path.join(package_dir, data_source)
        
        # Update data_source to point to full_path if it exists within the package
        if os.path.exists(full_path):
            data_source = full_path

    # Load the data into a DataFrame based on the type of data_source
    try:
        if isinstance(data_source, (dict, list)):
            # If the data_source is a dictionary or list, convert it to DataFrame directly
            original_data = pd.DataFrame(data_source)
        elif isinstance(data_source, str):
            ext = os.path.splitext(data_source)[-1].lower()
            
            # Load the file based on its extension
            if ext == '.csv':
                original_data = pd.read_csv(data_source)
            elif ext in ['.xls', '.xlsx']:
                original_data = pd.read_excel(data_source)
            elif ext == '.json':
                original_data = pd.read_json(data_source)
            elif ext == '.parquet':
                original_data = pd.read_parquet(data_source)
            elif ext == '.txt':
                original_data = pd.read_csv(data_source, delimiter='\t')
            elif ext == '.html':
                original_data = pd.read_html(data_source)[0]  # Assumes first table is relevant
            else:
                raise ValueError(f"Unsupported file format: {ext}. Supported formats are 'csv', 'xls', 'xlsx', 'json', 'parquet', 'txt', and 'html'.")
        else:
            raise ValueError("Unsupported data source. Provide a file path, dictionary, list, or DataFrame.")

        return original_data  # Return the DataFrame directly

    except Exception as e:
        raise RuntimeError(f"Failed to load data from {data_source}: {str(e)}")


def save(processed_data, filename, file_format='excel',include_change_log=False):
    """Save the processed DataFrame to a file in the specified format."""
    file_extension_map = {
        'csv': '.csv',
        'excel': '.xlsx',
        'json': '.json',
        'parquet': '.parquet',
        'txt': '.txt',
        'html': '.html'
    }

    # Validate file_format
    if file_format not in file_extension_map:
        raise ValueError(f"Unsupported file format: {file_format}. Choose from 'csv', 'excel', 'json', 'parquet', 'txt', or 'html'.")

    # Construct the full file path
    file_path = f"{filename}{file_extension_map[file_format]}"

    try:
        # Save the data in the specified format
        if file_format == 'csv':
            processed_data.to_csv(file_path, index=False)
        elif file_format == 'excel':
            processed_data.to_excel(file_path, index=False)
        elif file_format == 'json':
            processed_data.to_json(file_path, orient='records')
        elif file_format == 'parquet':
            processed_data.to_parquet(file_path)
        elif file_format == 'txt':
            processed_data.to_csv(file_path, index=False, sep='\t')  # Save as tab-delimited text file
        elif file_format == 'html':
            processed_data.to_html(file_path, index=False)

        # Confirm the save action
        if os.path.exists(file_path):
            print(f"File saved as: {file_path}")
            if include_change_log:
                print("Change log feature is removed in this version.")  # Placeholder message as change_log is removed

    except Exception as e:
        raise RuntimeError(f"Failed to save the file: {str(e)}")
