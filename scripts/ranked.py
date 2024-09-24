# Step 1: Install pandas (if not already installed)
# Open a terminal and run:
# pip install pandas

# Step 2: Import pandas
import pandas as pd

# Step 3: Load the CSV file
# Replace 'path/to/your/file.csv' with the actual path to your CSV file
df = pd.read_csv('/home/newtbeard/Downloads/project_choices.csv')

# Display the first few rows of the DataFrame
print(df.head())
