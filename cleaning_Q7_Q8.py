import pandas as pd
from collections import Counter


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