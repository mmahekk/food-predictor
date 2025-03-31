import re
import numpy as np
import pandas as pd
from io import StringIO


text_to_number = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10, 'eleven': 11, 'twelve': 12
}

word_bank_by_label = {
    'Pizza': {'cheese', 'tomato', 'dough', 'pepperoni', 'sauce', 'basil', 'water', 'oil', 'meat', 'olives'},
    'Shawarma': {'meat', 'pita', 'garlic', 'tahini', 'salad', 'onion', 'chicken', 'sauce', 'fries', 'rice'},
    'Sushi': {'rice', 'fish', 'seaweed', 'soy', 'wasabi', 'ginger', 'salmon', 'avocado'}
}

def parse_q2_response(response, word_bank=None):
    """
    Cleans a Q2 response by priority:
      1) Look for dash ranges (e.g., "3-4") and return their average.
      2) Look for remaining digits and return the largest found.
      3) Look for textual numbers (e.g., "three") and return the largest found.
      4) If none of the above are found, split the response on commas and count tokens
         that include at least one word from the provided word bank.
    """
    if not isinstance(response, str):
        return np.nan

    resp = response.lower().strip()
    numeric_values = []

    for low_str, high_str in re.findall(r'(\d+)-(\d+)', resp):
        avg_val = (int(low_str) + int(high_str)) / 2.0
        numeric_values.append(avg_val)

    resp = re.sub(r'\d+-\d+', '', resp)

    nums = re.findall(r'\b\d+\b', resp)
    if nums:
        numeric_values.append(max(int(num) for num in nums))

    words = re.findall(r'\b[a-z]+\b', resp)
    txt_nums = [text_to_number[word] for word in words if word in text_to_number]
    if txt_nums:
        numeric_values.append(max(txt_nums))

    if numeric_values:
        return float(max(numeric_values))
    else:
        tokens = [token.strip() for token in response.split(',')]
        count = 0
        for token in tokens:
            token_words = re.findall(r'\b[a-z]+\b', token.lower())
            if word_bank and any(word in word_bank for word in token_words):
                count += 1
        return float(count) if count > 0 else np.nan


# Example of how it will be used for Q2:
# data = """id,Q1,Q2: How many ingredients,Q3,Q4,Q5,Q6,Q7,Q8,Label
# 716549,3,6,"Lunch, Party","5","Movie1","Coke","Friends","Mild",Pizza
# 715742,4,"bread, meat","Lunch, Party","$5","Movie2","Coke","Friends, Teachers","None",Pizza
# 727333,3,5,"Lunch, Dinner","10","Movie3","Cola","Friends","Medium",Pizza
# 606874,4,"8, 6-7","Lunch, Dinner","$3","Movie4","Soda","Teachers","Hot",Pizza
# 700000,3,"bread, meat",Dinner,"$8","Movie5","Coke","Family","None",Pizza
# """
# df = pd.read_csv("/Users/mahek/Desktop/MISC/csc311-food-prediction/cleaned_data_combined.csv")

# def clean_q2_row(row):
#     bank = word_bank_by_label.get(row["Label"], None)
#     return parse_q2_response(row["Q2: How many ingredients would you expect this food item to contain?"], word_bank=bank)

# df["Q2_cleaned"] = df.apply(clean_q2_row, axis=1)

# print(df[["Q2: How many ingredients would you expect this food item to contain?", "Q2_cleaned", "Label"]])
