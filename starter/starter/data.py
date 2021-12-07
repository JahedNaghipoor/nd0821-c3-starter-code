
import pandas as pd
import os

def clean_data(df):
    df.replace({'?': None}, inplace=True)
    df.dropna(inplace=True)
    df.drop(["capital-gain", "capital-loss", "education-num", "fnlgt"], axis="columns", inplace=True)
    return df

def load_data(path):
    return pd.read_csv(path, skipinitialspace=True)

def save_data(df, path):
    return df.to_csv(path, index=False)

if __name__ == "__main__":
    dir = "../data/"
    load_path = os.path.join(dir, "census.csv")
    df = load_data(load_path)
    
    clean_df = clean_data(df)
    
    save_path = os.path.join(dir, "census_clean.csv")
    save_data(clean_df, save_path)
    print("Done")