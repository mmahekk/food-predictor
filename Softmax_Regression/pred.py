import pandas as pd
import numpy as np
import re


#functions from other files to use for cleaning data
mapping = {
    "cola": "coca-cola",
    "coke": "coca-cola",
    "coke zero": "coca-cola",
    "diet coke": "coca-cola",
    "diet pepsi": "pepsi",
    "crush": "fanta",
    "root beer": "soda",
    "soft drink": "soda",
    "carbonated water": "water",
    "soda pop": "soda",
    "pop": "soda",
    "bubble tea": "tea",
    "green tea": "tea",
    "black tea": "tea",
    "oolong tea": "tea",
    "jasmine tea": "tea",
    "barley tea": "tea",
    "hot tea": "tea",
    "matcha": "tea",
    "milk tea": "tea",
    "soy milk": "milk",
    "almond milk": "milk",
    "chocolate milk": "milk",
    "pineapple soda": "fanta",
    "miso soup": "soup",
    "fermented tea": "tea",
    "kombucha": "tea",
    "kraken rum": "rum",
    "martini cocktail": "cocktail",
    "calpis water": "water",
    "yakult drink": "yakult",
    "ramune soda": "soda",
    "rice wine": "sake",
    "nihonshu": "sake",
    "sparkling wine": "wine",
    "red wine": "wine",
    "white wine": "wine",
    "ginger beer": "beer",
    "spiced rum": "rum",
    "vodka cocktail": "cocktail",
    "gin martini": "martini",
    "water": None,
    "milk": None,
}

def map_drink(value):
    if pd.isna(value):
        return value
    value = value.lower().strip()
    for key in mapping.keys(): 
        if key in value:
            if mapping[key] is None:
                return value
            else:
                return mapping[key] 
    return value 

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

def process_q1_data(df):
    q1_col = [col for col in df.columns if 'Q1' in col][0]

    def extract_numeric(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float)):
            return float(value)
        matches = re.findall(r'\d+', str(value))
        if matches:
            return float(matches[0])
        return np.nan

    q1_values = df[q1_col].apply(extract_numeric)

    q1_features = np.array(q1_values).reshape(-1, 1)

    median_value = np.nanmedian(q1_features)
    q1_features = np.nan_to_num(q1_features, nan=median_value)

    return q1_features, "complexity_rating"


#modified to support not having a label column
def process_q2_data_fix(df):
    q2_col = [col for col in df.columns if 'Q2' in col][0]

    word_bank_by_label = {'cheese', 'tomato', 'dough', 'pepperoni', 'sauce', 'basil', 'water', 'oil', 'meat', 'olives', 'meat', 'pita', 'garlic', 'tahini', 'salad', 'onion', 'chicken', 'sauce', 'fries', 'rice', 'rice', 'fish', 'seaweed', 'soy', 'wasabi', 'ginger', 'salmon', 'avocado'},

    def clean_q2_row(row):
        return parse_q2_response(row[q2_col], word_bank=word_bank_by_label)

    q2_cleaned = df.apply(clean_q2_row, axis=1)

    q2_features = np.array(q2_cleaned).reshape(-1, 1)

    median_value = np.nanmedian(q2_features)
    q2_features = np.nan_to_num(q2_features, nan=median_value)

    return q2_features, "ingredient_count"


def process_q4_data_fix(df):
    q4_col = [col for col in df.columns if 'Q4' in col][0]

    def extract_price(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float)):
            return float(value)
        matches = re.findall(r'\d+(?:\.\d+)?', str(value))
        if matches:
            return float(max(matches, key=float))
        return np.nan

    q4_values = df[q4_col].apply(extract_price)

    q4_features = np.array(q4_values).reshape(-1, 1)

    median_value = np.nanmedian(q4_features)
    q4_features = np.nan_to_num(q4_features, nan=median_value)
    return q4_features, "price_expectation"


def softmax(z):
    """Compute softmax values for each set of scores in z."""
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True)) 
    return e_z / e_z.sum(axis=1, keepdims=True)

def predict(X, coef, intercept, classes):
    """Make predictions using the loaded model parameters."""
    #multiply by feature matrix and add bias
    scores = X @ coef.T + intercept
    #apply softmax to scores
    probabilities = softmax(scores)
    #get the predicted class
    predicted_indices = np.argmax(probabilities, axis=1)
    predicted_labels = classes[predicted_indices]
    return predicted_labels

#Note: this function is mostly reused code from the train_softmax_regression.py file
def create_features_for_prediction(df, params):
    """
    create feature matrix for prediction
    """
    features_list = []
    
    #process q1 (handle invalid inputs if they exist)
    q1_features, _ = process_q1_data(df.copy())
    features_list.append(q1_features)
    
    #process q2 into numbers (handle words in responses)
    temp_df_q2 = df.copy()
    q2_features, _ = process_q2_data_fix(temp_df_q2)
    features_list.append(q2_features)

    #process q3 into bag of words
    q3_col = [col for col in df.columns if 'Q3' in col][0]
    q3_word_to_idx = params['q3_word_to_idx'].item()
    n_q3_words = len(q3_word_to_idx)
    q3_features = np.zeros((len(df), n_q3_words))
    temp_q3 = df[q3_col].fillna('').str.lower().str.strip()
    for i, response in enumerate(temp_q3):
        if response:
            for word in response.split(','):
                word = word.strip()
                if word in q3_word_to_idx:
                    q3_features[i, q3_word_to_idx[word]] = 1
    features_list.append(q3_features)

    #process q6 into bag of words
    q6_col = [col for col in df.columns if 'Q6' in col][0]
    q6_word_to_idx = params['q6_word_to_idx'].item()
    n_q6_words = len(q6_word_to_idx)
    q6_features = np.zeros((len(df), n_q6_words))
    temp_q6 = df[q6_col].fillna('').str.lower().str.strip()
    temp_q6 = temp_q6.apply(map_drink) 
    for i, response in enumerate(temp_q6):
         if response:
            for word in response.split(','):
                word = word.strip()
                if word in q6_word_to_idx:
                    q6_features[i, q6_word_to_idx[word]] = 1
    features_list.append(q6_features)

    #process q4 into numbers
    q4_features, _ = process_q4_data_fix(df.copy())
    features_list.append(q4_features)

    #process q5 into bag of words
    q5_col = [col for col in df.columns if 'Q5' in col][0]
    q5_word_to_idx = params['q5_word_to_idx'].item()
    n_q5_words = len(q5_word_to_idx)
    q5_features = np.zeros((len(df), n_q5_words))
    temp_q5 = df[q5_col].fillna('').str.lower().str.strip()
    for i, response in enumerate(temp_q5):
         if response:
            for movie in response.split(','):
                movie = movie.strip()
                if movie in q5_word_to_idx:
                    q5_features[i, q5_word_to_idx[movie]] = 1
    features_list.append(q5_features)

    q7_word_to_idx = params['q7_word_to_idx'].item()
    q8_category_to_idx = params['q8_category_to_idx'].item()
    n_q7_words = len(q7_word_to_idx)
    n_q8_cats = len(q8_category_to_idx)
    q7_features = np.zeros((len(df), n_q7_words))
    q8_features = np.zeros((len(df), n_q8_cats))
    
    q7_col = [col for col in df.columns if 'Q7' in col][0]
    q8_col = [col for col in df.columns if 'Q8' in col][0]
    temp_q7 = df[q7_col].fillna('').str.lower().str.strip()
    temp_q8 = df[q8_col].fillna('').str.lower().str.strip()

    #process q7 into bag of words
    for i, response in enumerate(temp_q7):
        if response:
            for word in response.split(','): 
                word = word.strip()
                if word in q7_word_to_idx:
                    q7_features[i, q7_word_to_idx[word]] = 1

    #process q8 into one hot
    for i, response in enumerate(temp_q8):
         if response in q8_category_to_idx:
            q8_features[i, q8_category_to_idx[response]] = 1

    features_list.append(q7_features)
    features_list.append(q8_features)

    X = np.hstack(features_list)
        
    return X

def predict_all(csv_file_path):
    """
    predict all datapoints in csv file
    """

    #load params in
    params = np.load("softmax_model_params.npz", allow_pickle=True) 
        
    coef = params['coef']
    intercept = params['intercept']
    classes = params['classes']

    #load the csv file into a dataframe
    df_predict = pd.read_csv(csv_file_path)

    
    X_predict = create_features_for_prediction(df_predict, params)

    predictions = predict(X_predict, coef, intercept, classes)

    return predictions

