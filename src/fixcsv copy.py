import pandas as pd
import ast
import csv

# Define conversion function
def convert_list(x):
    try:
        return ast.literal_eval(x)
    except Exception:
        return x

# Specify columns that should be converted to lists
converters = {
    'agent_1_scores': convert_list,
    'agent_2_scores': convert_list,
    'agent_1_messages': convert_list,
    'agent_2_messages': convert_list,
    'agent_1_actions': convert_list,
    'agent_2_actions': convert_list
}

# Read the CSV with converters
df = pd.read_csv('/cluster/home/mlindner/Github/master_thesis_project/src/data/outputs/experiment_250225_fixed.csv', converters=converters)

# Save back to CSV without extra quotes
df.to_csv('/cluster/home/mlindner/Github/master_thesis_project/src/data/outputs/experiment_250225_fixed_fixed.csv', index=False, quoting=csv.QUOTE_NONE, escapechar=' ')
