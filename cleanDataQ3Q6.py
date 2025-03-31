import pandas as pd


# df = pd.read_csv("cleaned_data_combined.csv")
# # print(df['Q6: What drink would you pair with this food item?'].unique())


# df['Q6: What drink would you pair with this food item?'] = df['Q6: What drink would you pair with this food item?'].str.lower()
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
    if pd.isna(value):  # Handle NaN values
        return value
    value = value.lower().strip()  # Normalize case and spaces
    for key in mapping.keys():  # Check for key presence in the text
        if key in value:
            if mapping[key] is None:
                return value
            else:
                return mapping[key]  # Replace with mapped value
    return value  # Keep original if no match found

# check = df['Q6: What drink would you pair with this food item?'].tolist()
# print(check)
# df['Q6: What drink would you pair with this food item?'] = df['Q6: What drink would you pair with this food item?'].apply(map_drink)
# # df['Q6: What drink would you pair with this food item?'] = df['Q6: What drink would you pair with this food item?'].replace(mapping)
# # df['Q6: What drink would you pair with this food item?'] = df['Q6: What drink would you pair with this food item?'].str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
# # df['Q6: What drink would you pair with this food item?'] = df['Q6: What drink would you pair with this food item?'].str.replace("\xa0", "", regex=True)
# unique_cleaned_values = df['Q6: What drink would you pair with this food item?'].tolist()
# print(unique_cleaned_values)

# def clean_text(text):
#     """Removes extra spaces, special characters, and standardizes capitalization."""
#     if isinstance(text, str):
#         text = text.strip().lower()
#         text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
#     return text.capitalize()  # Capitalize first letter
#
#
#     drink_mapping = {
#         "cola": "Coke",
#         "coke": "Coke",
#         "coke ": "Coke",  # Handle trailing space issue
#         "soda": "Soda",
#     }
#
#
# def extract_numbers(text):
#     """Extracts numeric values from a text string."""
#     if isinstance(text, str):
#         match = re.findall(r'\d+', text)  # Find all numbers
#         return " ".join(match) if match else ""
#     return text
#
#
# def clean_data(input_file, output_file):
#     # Load dataset
#     df = pd.read_csv(input_file)
#
#     # Clean Q6 (Drink Pairing) Responses
#     df["Q6_Cleaned"] = df["Q6"].apply(clean_text)
#     df["Q6_Standardized"] = df["Q6_Cleaned"].apply(standardize_drink)
#
#     # Standardize Q2 (Ingredients Count)
#     df["Q2_Cleaned"] = df["Q2"].apply(extract_numbers)
#
#     # Standardize Q4 (Price Expectation)
#     df["Q4_Cleaned"] = df["Q4"].apply(extract_numbers)
#
#     # Save cleaned data
#     df.to_csv(output_file, index=False)
#     print(f"Cleaned data saved to {output_file}")
