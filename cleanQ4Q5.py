import pandas as pd
import numpy as np
import re

# Load dataset
data = pd.read_csv("cleaned_data_combined_modified.csv")

# QUESTION 4 - PRICE QUESTION

currency_symbols = ["$", "CAD", "USD", "dollars", "dollar", "yen", "¥", "€", "£"]

# Function to clean and extract prices
def extract_prices(text):
    if pd.isna(text) or text.strip() == "":
        return np.nan, np.nan  # Empty values
    
    original_text = text.lower()

    # Remove currency words but keep numbers
    for currency in currency_symbols:
        original_text = original_text.replace(currency, "").strip()

    # Extract numbers
    numbers = re.findall(r"\d+(?:-\d+)?(?:\.\d+)?", original_text)  # Captures ranges (e.g., 20-30)
    
    if not numbers:
        return np.nan, np.nan  # No numbers found
    
    # Process ranges like '20-30'
    processed_numbers = []
    for num in numbers:
        if '-' in num:
            lower, upper = num.split('-')
            processed_numbers.append((float(lower) + float(upper)) / 2)  # Average of range
        else:
            processed_numbers.append(float(num))
    
    # Assume the first number is the full price (if there's no unit specified)
    full_price = np.nan
    if len(processed_numbers) == 1:
        full_price = processed_numbers[0]
    elif len(processed_numbers) > 1:
        # If multiple numbers, return the average of all numbers as full price
        full_price = np.mean(processed_numbers)

    return full_price

# Apply the function to Q4 column
data['Q4_Full_Price'] = data['Q4: How much would you expect to pay for one serving of this food item?'].apply(extract_prices)

# QUESTION 5 - MOVIE QUESTION

def create_movie_features(df):
    # Extract all unique movies from the responses
    all_movies = set()
    for response in df["Q5: What movie do you think of when thinking of this food item?"].dropna():
        movies = response.split(",")
        for movie in movies:
            all_movies.add(movie.strip().lower())  # Store movies in lowercase for consistency
    
    all_movies = sorted(list(all_movies))  # Sorted list of unique movies
    movie_to_idx = {movie: idx for idx, movie in enumerate(all_movies)}  # Map movies to index
    
    # Initialize an empty matrix (rows: samples, columns: movies)
    movie_features = np.zeros((len(df), len(all_movies)))
    
    for i, response in enumerate(df["Q5: What movie do you think of when thinking of this food item?"].fillna('')):
        if response:
            movies = response.split(",")
            for movie in movies:
                movie = movie.strip().lower()
                if movie in movie_to_idx:
                    movie_features[i, movie_to_idx[movie]] = 1
    
    return movie_features, movie_to_idx

# Create the movie feature matrix
movie_features, movie_to_idx = create_movie_features(data)

# Convert to DataFrame for better readability
movie_features_df = pd.DataFrame(movie_features, columns=movie_to_idx.keys())

# Combine the cleaned price data and movie features with the original dataset
# Join the cleaned data to the original dataframe
final_data = pd.concat([data, movie_features_df], axis=1)

# final_data.to_csv("cleaned_data_q4_q5_combined.csv", index=False)

# Show the final cleaned data
print(final_data.head())









