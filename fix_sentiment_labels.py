"""
Fix zuco_sentiment_labels_task1.csv (same fixes as in Thesis-Data structures-ZuCo.ipynb):

1. Drop the leading comment row ("# -1 = negative, 0 = neutral, 1 = positive"),
   identified by `control` being empty/None.
2. Fix the 5 rows where `sentiment_label` and `control` got swapped
   (sentiment_label is NaN but control holds the real -1/0/1 value).
3. Patch 2 sentence texts that have stray digit characters in the ZuCo .mat
   `content` field but not in this CSV, so text-based matching against the
   .mat files succeeds (found via diffing against resultsZAB_SR.mat).
4. Cast sentence_id and sentiment_label to int.
5. Save the cleaned CSV.

Run this in Colab (paths point at your Drive), or locally with the paths adjusted.
"""

import pandas as pd
import numpy as np

RAW_CSV = '/content/drive/MyDrive/Parmis/Thesis/Data/zuco_sentiment_labels_task1.csv'
FIXED_CSV = '/content/drive/MyDrive/Parmis/Thesis/Data/zuco_sentiment_labels_task1_fixed.csv'

df = pd.read_csv(RAW_CSV, sep=None, engine='python')

# 1. drop the comment row (control is exactly None there, not NaN -
#    most real rows have an empty/NaN control too, so isna() would be too broad)
none_rows = df[df['control'].apply(lambda x: x is None)]
print('dropping rows:')
print(none_rows)
df = df[~df['control'].apply(lambda x: x is None)].copy()

# 2. fix the rows where sentiment_label and control got swapped
faulty_mask = df['sentiment_label'].isna()
print(f'fixing {faulty_mask.sum()} swapped rows')
df.loc[faulty_mask, 'sentiment_label'] = df.loc[faulty_mask, 'control'].astype(float)
df.loc[faulty_mask, 'control'] = np.nan

# 3. patch sentence texts to match the ZuCo .mat 'content' field exactly
TEXT_FIXES = {
    "Bullock's complete lack of focus and ability quickly derails the film.":
        "Bullock's complete lack of focus and ability quickly derails the film.1",
    "Ultimately feels empty and unsatisfying, like swallowing a Communion wafer without the wine.":
        "Ultimately feels emp11111ty and unsatisfying, like swallowing a Communion wafer without the wine.",
}
df['sentence'] = df['sentence'].replace(TEXT_FIXES)

# 4. cast types
df['sentence_id'] = df['sentence_id'].astype(int)
df['sentiment_label'] = df['sentiment_label'].astype(int)

# 5. save
print(df.shape)
print(df['sentiment_label'].value_counts())

df.to_csv(FIXED_CSV, index=False)
print('saved to', FIXED_CSV)
