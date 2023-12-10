import pandas as pd
import os

# Load your CSV file into a DataFrame
df = pd.read_csv(os.path.join('data','census.csv'))

# Clean headers by stripping spaces
df.columns = [col.lstrip() for col in df.columns]

# Apply str.lstrip to remove leading spaces from each entry
df = df.applymap(lambda x: x.lstrip() if isinstance(x, str) else x)

# Optionally, save the cleaned DataFrame back to a CSV file
df.to_csv(os.path.join('data','cleaned_census.csv'), index=False)