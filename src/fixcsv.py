import pandas as pd
import ast
import re
import csv

def clean_list_column(value):
    """Fixes improperly formatted lists with inconsistent quotes."""
    if not (isinstance(value, str) and value.startswith('[') and value.endswith(']')):
        return value

    # Remove outer quotes if the whole list is enclosed, e.g. "\"[ ... ]\""
    if value.startswith('"[') and value.endswith(']"'):
        value = value[1:-1]
    
    # Replace occurrences of double double-quotes with a single double quote.
    value = value.replace('""', '"')
    
    # Fix contraction issues: e.g. turns "Let"s into "Let's"
    value = re.sub(r'"([A-Za-z]+)"s', r'"\1\'s', value)
    
    try:
        # Attempt to evaluate the string as a Python literal.
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        # Fallback: extract quoted substrings using regex.
        pattern = r'(["\'])(.*?)(?<!\\)\1'
        items = re.findall(pattern, value)
        if items:
            return [item[1] for item in items]
    return value

def list_to_str(x):
    if isinstance(x, list):
        # If all elements are numbers (int or float), output them without quotes.
        if all(isinstance(item, (int, float)) for item in x):
            return '[' + ', '.join(str(item) for item in x) + ']'
        else:
            # Otherwise, assume they are strings and enclose each in double quotes.
            return '[' + ', '.join(f'"{item}"' for item in x) + ']'
    return x

def fix_csv(file_path, output_path):
    df = pd.read_csv(file_path, dtype=str)
    
    # Define columns expected to contain list data.
    list_columns = [
        'agent_1_messages', 'agent_2_messages', 
        'agent_1_actions', 'agent_2_actions',
        'agent_1_scores', 'agent_2_scores'
    ]
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_list_column)
    
    # Convert list objects back into the desired string format.
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(list_to_str)
    
    # Write the cleaned CSV using QUOTE_MINIMAL to avoid extra quoting.
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)

fix_csv("/cluster/home/mlindner/Github/master_thesis_project/src/data/outputs/experiment_250225.csv", "/cluster/home/mlindner/Github/master_thesis_project/src/data/outputs/experiment_250225_fixed.csv")
