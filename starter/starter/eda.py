
#%%
# Add code to load in the data.
import pandas as pd
def load_data():
    return pd.read_csv('../data/census.csv')

#%%  
df = load_data()
df.head()
# %%
df.columns = df.columns.str.replace(' ', '')

# %%
df.to_csv('../data/census_clean.csv')

