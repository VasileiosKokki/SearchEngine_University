import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_all_docs_metadata(file_path_docs):
    ext = os.path.splitext(file_path_docs)[1].lower()
    delimiter = ',' if ext == '.csv' else '\t'

    # Read full file with pandas
    df = pd.read_csv(file_path_docs, delimiter=delimiter, encoding='utf-8', low_memory=False)

    # Add auto-increment doc_id as string like "doc_0", "doc_1", ...
    df['doc_id'] = ['doc_' + str(i) for i in range(len(df))]

    # Replace 'unknown' and similar with NaN
    df = df.applymap(replace_unknown_with_nan)

    # Check for missing values
    missing_values = df.isnull().sum()  # Count missing values per column

    # Print missing or empty values
    for column in df.columns:
        if missing_values[column] > 0:
            print(f"Column '{column}' has {missing_values[column]} missing values")

    # Drop exact duplicates based on Title + Release Year
    before = df.shape[0]
    df = df.drop_duplicates(subset=['Title', 'Release Year'], keep='first')
    after = df.shape[0]
    print(f"Dropped {before - after} duplicate rows based on Title and Release Year.")

    docs_metadata = {}

    # Iterate rows efficiently
    for _, row in tqdm(df.iterrows(), total=len(df), unit="lines", desc="Loading docs"):
        # Make sure enough columns (safety)
        if len(row) <= 7:
            continue

        doc_id = row['doc_id']
        release_year = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else np.nan
        title = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else np.nan
        origin = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else np.nan
        director = str(row.iloc[3]).strip() if pd.notna(row.iloc[3]) else np.nan
        cast = str(row.iloc[4]).strip() if pd.notna(row.iloc[4]) else np.nan
        genre = str(row.iloc[5]).strip() if pd.notna(row.iloc[5]) else np.nan
        url = str(row.iloc[6]).strip() if pd.notna(row.iloc[6]) else np.nan
        plot = str(row.iloc[7]).strip() if pd.notna(row.iloc[7]) else np.nan

        evaluation_parts = [title, plot, genre]
        evaluation = " ".join([part for part in evaluation_parts if pd.notna(part)])

        docs_metadata[doc_id] = {
            'release_year': release_year,
            'title': title,
            'origin': origin,
            'director': director,
            'cast': cast,
            'genre': genre,
            'url': url,
            'plot': plot,
            'evaluation': evaluation
        }

    return docs_metadata

def replace_unknown_with_nan(value):
    if pd.isnull(value) or str(value).strip() == "" or str(value).strip().lower() == "unknown":
        return np.nan  # Replace with NaN
    return value  # Otherwise, return the value as is


def load_stopwords():
    stopwords = []
    with open('stopwords.large', 'r', encoding='utf-8') as f:
        for word in f:
            stopwords.append(word.strip().lower())
    return stopwords