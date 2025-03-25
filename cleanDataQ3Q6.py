import pandas as pd
from io import StringIO
import re
df = pd.read_csv("cleaned_data_combined.csv")
# print(df['Q6: What drink would you pair with this food item?'].unique())


df['Q6: What drink would you pair with this food item?'] = df['Q6: What drink would you pair with this food item?'].str.lower()
mapping = {
    "cola": "coca-cola",
    "coke": "coca-cola",
    "coke zero": "coca-cola",
    "diet coke": "coca-cola",
    "soda": "soda",
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
    "water": "water",
    "milk": "milk",
    "coca-cola": "coca-cola",
    "tea": "tea",
    "juice": "juice",
    "sake": "sake",
    "soup": "soup",
    "soju": "soju",
    "alcohol": "alcohol",
    "coca cola": "coca-cola",
    "pepsi": "pepsi", "sprite": "sprite", "fanta": "fanta", "7up": "7up",
    "ginger ale": "ginger ale", "canada dry": "canada dry ginger ale",
    "iced tea": "iced tea", "boba": "bubble tea", "coffee": "coffee",
    "sparkling water": "sparkling water", "mineral water": "sparkling water",
    "lemonade": "lemonade", "orange juice": "orange juice",
    "apple juice": "apple juice", "mango juice": "mango juice",
    "pineapple juice": "pineapple juice", "lassi": "lassi",
    "beer": "beer",
    "wine": "wine", "whiskey": "whiskey",
    "cocktail": "cocktail", "martini": "martini",
    "kraken spiced rum": "rum", "rum": "rum", "calpis": "calpis",
    "yakult": "yakult", "ramune": "ramune", "powerade": "powerade",
    "gatorade": "gatorade", "mountain dew": "mountain dew",
    "dr pepper": "dr pepper", "ayran": "ayran", "barbican": "barbican",
    "champagne": "champagne", "leban": "leban", "mint lemonade": "mint lemonade",
    "smoothie": "smoothie", "mango lassi": "mango lassi",
    "mango pulp": "mango pulp", "diet brisk": "diet brisk",
    "saporo": "saporo", "soy sauce": "soy sauce", "baijiu": "baijiu",
    "ocha": "ocha"
}




# function uses mapping to check if keyword is in mapping, if it is, then you either return the value associated
# or return the keyword itself
def map_drink(value):
    if pd.isna(value):  # Handle NaN values
        return value
    value = re.sub(r'[^a-zA-Z0-9\s]', '', value.lower().strip())
    for key in mapping.keys():  # Check for key presence in the text
        if key in value:
            return mapping[key]  # Replace with mapped value

    print(value)
    return value  # Keep original if no match found


df['Q6: What drink would you pair with this food item?'] = df['Q6: What drink would you pair with this food item?'].apply(map_drink)
# df['Q6: What drink would you pair with this food item?'] = df['Q6: What drink would you pair with this food item?'].replace(mapping)
# df['Q6: What drink would you pair with this food item?'] = df['Q6: What drink would you pair with this food item?'].str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
# df['Q6: What drink would you pair with this food item?'] = df['Q6: What drink would you pair with this food item?'].str.replace("\xa0", "", regex=True)
unique_cleaned_values = df['Q6: What drink would you pair with this food item?'].unique().tolist()
# for value in unique_cleaned_values:
#     if value not in mapping:
#         # print(f"Value {value} is not in the mapping")
print(unique_cleaned_values)


# Example of how it will be used for Q2:
data = """id,Q1,Q2,Q3,Q4,Q5,Q6: What drink would you pair with this food item?,Q7,Q8,Label
716549,3,6,"Lunch, Party","5","Movie1","Coke","Friends","Mild",Pizza
715742,4,"bread, meat","Lunch, Party","$5","Movie2","Coke","Friends, Teachers","None",Pizza
727333,3,5,"Lunch, Dinner","10","Movie3","Cola","Friends","Medium",Pizza
606874,4,"8, 6-7","Lunch, Dinner","$3","Movie4","Soda","Teachers","Hot",Pizza
700000,3,"bread, meat",Dinner,"$8","Movie5","Coke","Family","None",Pizza
"""

# process the test data to a pandas dataframe
df_test = pd.read_csv(StringIO(data))

# Applies the map_drink function on each column (axis = 0 is set as default) where axis = 1 means that the
# entire row is sent and the function is applied on the entire row.
df_test["Q6_cleaned"] = df_test['Q6: What drink would you pair with this food item?'].apply(map_drink)
# Print output
print(df_test[['Q6: What drink would you pair with this food item?', "Q6_cleaned", "Label"]])



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
