import pandas as pd
from collections import Counter
import numpy as np


def clean_data(file_path):
    df = pd.read_csv(file_path)

    df["Q7: When you think about this food item, who does it remind you of?"] = (
        df["Q7: When you think about this food item, who does it remind you of?"]
        .str.lower()
        .str.replace(r"[^a-z, ]", "", regex=True) 
        .str.strip()
    )

    df["Q8: How much hot sauce would you add to this food item?"] = df[
        "Q8: How much hot sauce would you add to this food item?"
    ].str.strip().fillna("None")  

    #count responses for Q7
    q7_counts = df["Q7: When you think about this food item, who does it remind you of?"].value_counts()

    #count responses for Q8
    df["Q8: How much hot sauce would you add to this food item?"].value_counts()


    #get counts of each index by the target value (shawarma, pizza, sushi)
    q7_by_label = df.groupby("Label")["Q7: When you think about this food item, who does it remind you of?"].value_counts().unstack().fillna(0)
    q8_by_label = df.groupby("Label")["Q8: How much hot sauce would you add to this food item?"].value_counts().unstack().fillna(0)


    #split Q7 responses by comma and make each word a seperate count
    q7_word_counts = {}
    for label, group in df.groupby("Label")["Q7: When you think about this food item, who does it remind you of?"]:
        words = []
        for response in group.dropna():
            words.extend(response.split(","))
        word_count = Counter(words)
        q7_word_counts[label] = word_count


    q7_word_df = pd.DataFrame(q7_word_counts).fillna(0).astype(int)
    return (q7_word_df, q8_by_label, q7_by_label, q7_counts, df)


def process_data(df):
    #Q7 process into bag of words
    all_words = set()
    for response in df["Q7: When you think about this food item, who does it remind you of?"].dropna():
        words = response.split(',')
        all_words.update(word.strip() for word in words)
    
    n_samples = len(df)
    n_words = len(all_words)
    word_to_idx = {word: idx for idx, word in enumerate(sorted(all_words))}
    
    q7_features = np.zeros((n_samples, n_words))
    for i, response in enumerate(df["Q7: When you think about this food item, who does it remind you of?"].fillna('')):
        if response:
            words = response.split(',')
            for word in words:
                word = word.strip()
                if word:
                    q7_features[i, word_to_idx[word]] = 1
                        
    #Q8 one hot encoding
    hot_sauce_levels = ['None', 'A little (mild)', 'A moderate amount (medium)', 'A lot (hot)', 'I will have some of this food item with my hot sauce']
    hot_sauce_to_idx = {level: idx for idx, level in enumerate(hot_sauce_levels)}
    
    q8_features = np.zeros((n_samples, len(hot_sauce_levels)))
    for i, response in enumerate(df["Q8: How much hot sauce would you add to this food item?"]):
        if response in hot_sauce_to_idx:
            q8_features[i, hot_sauce_to_idx[response]] = 1
    
    return q7_features.astype(int), q8_features.astype(int), word_to_idx, hot_sauce_to_idx





