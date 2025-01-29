import pandas as pd
import re

# 1. Load the Excel file containing the text
input_file_path = "C:\\py\\Last\\DSET.xlsx"  # Replace with your input file path
output_file_path = "C:\\py\\Last\\split_words.xlsx"  # Output file path
column_name = "tw"  # Replace with the actual column name in your Excel file

# Read the Excel file
data = pd.read_excel(input_file_path)

# Ensure the column exists
if column_name not in data.columns:
    raise ValueError(f"Column '{column_name}' not found in the Excel file.")

# 2. Process the text: Split into words
all_words = []

for text in data[column_name].dropna():  # Skip missing values
    words = re.findall(r'\b\w+\b', str(text))  # Extract words using regex
    all_words.extend(words)

# 3. Save the words to a new Excel file
df_words = pd.DataFrame({'term': all_words})
df_words.to_excel(output_file_path, index=False)

print(f"Words have been split and saved to: {output_file_path}")


# Steps to Use the Code:
# Prepare Your Input File:
# Ensure your input Excel file (e.g., input_texts.xlsx) contains a column named text with the tweets or text you want to process. Example:
# text
# Hello world, Python rocks!
# Learning is fun.
# AI is changing the world.
# Run the Code:

# Replace input_texts.xlsx with the path to your input file.
# Replace text with the name of the column in your file if it's different.
# Output File:

# The output file (split_words.xlsx) will contain each word from the input file as a row. Example:
