import pandas as pd
import numpy as np
import re

data = pd.read_csv("cleaned_data_combined_modified.csv")

# QUESTION 4 - PRICE QUESTION

currency_symbols = ["$", "CAD", "USD", "dollars", "dollar", "yen", "¥", "€", "£"]

def extract_prices(text):
    if pd.isna(text) or text.strip() == "":
        return np.nan, np.nan 
    
    original_text = text.lower()

    # remove words
    for currency in currency_symbols:
        original_text = original_text.replace(currency, "").strip()

    numbers = re.findall(r"\d+(?:-\d+)?(?:\.\d+)?", original_text)  #(e.g., 20-30)
    
    if not numbers:
        return np.nan, np.nan  # no numbers found
    
    # process ranges like '20-30'
    processed_numbers = []
    for num in numbers:
        if '-' in num:
            lower, upper = num.split('-')
            processed_numbers.append((float(lower) + float(upper)) / 2)  # avg
        else:
            processed_numbers.append(float(num))
    
    # assume the first number is the full price (if there's no unit specified)
    full_price = np.nan
    if len(processed_numbers) == 1:
        full_price = processed_numbers[0]
    elif len(processed_numbers) > 1:
        # if multiple numbers, return the average
        full_price = np.mean(processed_numbers)

    return full_price

data['Q4_Full_Price'] = data['Q4: How much would you expect to pay for one serving of this food item?'].apply(extract_prices)

# QUESTION 5 - MOVIE QUESTION

def create_movie_features(df):
    # Extract all unique movies from the responses
    all_movies = set()
    for response in df["Q5: What movie do you think of when thinking of this food item?"].dropna():
        movies = response.split(",")
        for movie in movies:
            all_movies.add(movie.strip().lower())  # store movies in lowercase for consistency
    
    all_movies = sorted(list(all_movies))  # sorted list of unique movies
    movie_to_idx = {movie: idx for idx, movie in enumerate(all_movies)}  # map movies to index
    
    movie_features = np.zeros((len(df), len(all_movies)))
    
    for i, response in enumerate(df["Q5: What movie do you think of when thinking of this food item?"].fillna('')):
        if response:
            movies = response.split(",")
            for movie in movies:
                movie = movie.strip().lower()
                if movie in movie_to_idx:
                    movie_features[i, movie_to_idx[movie]] = 1
    
    return movie_features, movie_to_idx









