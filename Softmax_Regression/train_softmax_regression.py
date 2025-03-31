import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, log_loss
import sys
import re

sys.path.append('../')

from Questions_7_8.cleaning_Q7_Q8 import process_data
from Questions_1_2.cleaning_Q1_Q2 import parse_q2_response
from cleanDataQ3Q6 import map_drink
from Questions_4_5.cleanQ4Q5 import create_movie_features


def process_q3q6_data(df):
    """
    Process Q3 (meal occasion) and Q6 (drink pairing) into bag of words features.

    Args:
        df: DataFrame containing the survey data

    Returns:
        q3_features: numpy array of shape (n_samples, n_q3_words)
        q6_features: numpy array of shape (n_samples, n_q6_words)
        q3_word_to_idx: dictionary mapping Q3 words to column indices
        q6_word_to_idx: dictionary mapping Q6 drinks to column indices
    """
    q3_col = [col for col in df.columns if 'Q3' in col][0]
    df.loc[:, q3_col] = df[q3_col].fillna('').str.lower().str.strip()

    q6_col = [col for col in df.columns if 'Q6' in col][0]
    df.loc[:, q6_col] = df[q6_col].fillna('').str.lower().str.strip()

    df.loc[:, q6_col] = df[q6_col].apply(map_drink)

    all_q3_words = set()
    for response in df[q3_col].dropna():
        words = response.split(',')
        all_q3_words.update(word.strip() for word in words if word.strip())

    q3_word_to_idx = {word: idx for idx, word in enumerate(sorted(all_q3_words))}

    all_q6_words = set()
    for response in df[q6_col].dropna():
        words = response.split(',')
        all_q6_words.update(word.strip() for word in words if word.strip())

    q6_word_to_idx = {word: idx for idx, word in enumerate(sorted(all_q6_words))}

    n_samples = len(df)
    n_q3_words = len(q3_word_to_idx)
    n_q6_words = len(q6_word_to_idx)

    q3_features = np.zeros((n_samples, n_q3_words))
    q6_features = np.zeros((n_samples, n_q6_words))

    for i, response in enumerate(df[q3_col].fillna('')):
        if response:
            words = response.split(',')
            for word in words:
                word = word.strip()
                if word and word in q3_word_to_idx:
                    q3_features[i, q3_word_to_idx[word]] = 1

    for i, response in enumerate(df[q6_col].fillna('')):
        if response:
            words = response.split(',')
            for word in words:
                word = word.strip()
                if word and word in q6_word_to_idx:
                    q6_features[i, q6_word_to_idx[word]] = 1

    return q3_features, q6_features, q3_word_to_idx, q6_word_to_idx


def load_data(file_path):
    return pd.read_csv(file_path)


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


def process_q2_data_fix(df):
    q2_col = [col for col in df.columns if 'Q2' in col][0]

    word_bank_by_label = {
        'Pizza': {'cheese', 'tomato', 'dough', 'pepperoni', 'sauce', 'basil', 'water', 'oil', 'meat', 'olives'},
        'Shawarma': {'meat', 'pita', 'garlic', 'tahini', 'salad', 'onion', 'chicken', 'sauce', 'fries', 'rice'},
        'Sushi': {'rice', 'fish', 'seaweed', 'soy', 'wasabi', 'ginger', 'salmon', 'avocado'}
    }

    def clean_q2_row(row):
        bank = word_bank_by_label.get(row["Label"], None)
        return parse_q2_response(row[q2_col], word_bank=bank)

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


def train_softmax_regression_model(df_train, df_test):
    train_features_list = []
    test_features_list = []
    feature_names = []

    # Q1: Complexity
    q1_train_features, q1_name = process_q1_data(df_train)
    q1_test_features, _ = process_q1_data(df_test)
    train_features_list.append(q1_train_features)
    test_features_list.append(q1_test_features)
    feature_names.append(q1_name)

    q2_train_features, q2_name = process_q2_data_fix(df_train)
    q2_test_features, _ = process_q2_data_fix(df_test)
    train_features_list.append(q2_train_features)
    test_features_list.append(q2_test_features)
    feature_names.append(q2_name)

    q3_train_features, q6_train_features, q3_word_to_idx, q6_word_to_idx = process_q3q6_data(df_train)
    q3_test_features = np.zeros((len(df_test), len(q3_word_to_idx)))
    q6_test_features = np.zeros((len(df_test), len(q6_word_to_idx)))

    q3_col = [col for col in df_test.columns if 'Q3' in col][0]
    q6_col = [col for col in df_test.columns if 'Q6' in col][0]

    for i, response in enumerate(df_test[q3_col].fillna('')):
        if response:
            for word in response.split(','):
                word = word.strip().lower()
                if word in q3_word_to_idx:
                    q3_test_features[i, q3_word_to_idx[word]] = 1

    for i, response in enumerate(df_test[q6_col].fillna('')):
        if response:
            for word in response.split(','):
                word = word.strip().lower()
                if word in q6_word_to_idx:
                    q6_test_features[i, q6_word_to_idx[word]] = 1

    train_features_list.append(q3_train_features)
    train_features_list.append(q6_train_features)
    test_features_list.append(q3_test_features)
    test_features_list.append(q6_test_features)
    feature_names.extend([f'q3_{word}' for word in sorted(q3_word_to_idx.keys())])
    feature_names.extend([f'q6_{word}' for word in sorted(q6_word_to_idx.keys())])

    q4_train_features, q4_name = process_q4_data_fix(df_train)
    q4_test_features, _ = process_q4_data_fix(df_test)
    train_features_list.append(q4_train_features)
    test_features_list.append(q4_test_features)
    feature_names.append(q4_name)

    q5_train_features, q5_word_to_idx = create_movie_features(df_train)
    q5_test_features = np.zeros((len(df_test), len(q5_word_to_idx)))
    q5_col = [col for col in df_test.columns if 'Q5' in col][0]
    for i, response in enumerate(df_test[q5_col].fillna('')):
        if response:
            for movie in response.split(','):
                movie = movie.strip().lower()
                if movie in q5_word_to_idx:
                    q5_test_features[i, q5_word_to_idx[movie]] = 1
    train_features_list.append(q5_train_features)
    test_features_list.append(q5_test_features)
    feature_names.extend([f'q5_{movie}' for movie in sorted(q5_word_to_idx.keys())])

    q7_train_features, q8_train_features, q7_word_to_idx, q8_category_to_idx = process_data(df_train)
    q7_test_features, q8_test_features, _, _ = process_data(df_test)
    train_features_list.append(q7_train_features)
    train_features_list.append(q8_train_features)
    test_features_list.append(q7_test_features)
    test_features_list.append(q8_test_features)
    feature_names.extend([f'q7_{word}' for word in sorted(q7_word_to_idx.keys())])
    feature_names.extend([f'q8_{cat}' for cat in sorted(q8_category_to_idx.keys())])

    X_train = np.hstack(train_features_list)
    X_test = np.hstack(test_features_list)

    y_train = df_train['Label'].values
    y_test = df_test['Label'].values

    # Note: multi_class defaults to "multinomial" in future versions, so we can omit it if desired.
    softmax_model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
    softmax_model.fit(X_train, y_train)

    y_train_pred = softmax_model.predict(X_train)
    y_test_pred = softmax_model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred)

    train_probs = softmax_model.predict_proba(X_train)
    test_probs = softmax_model.predict_proba(X_test)
    train_loss = log_loss(y_train, train_probs)
    test_loss = log_loss(y_test, test_probs)

    return test_accuracy, train_accuracy, report, softmax_model, feature_names, X_train.shape, train_loss, test_loss


def main():
    full_data_path = "../cleaned_data_combined.csv"
    df = pd.read_csv(full_data_path)
    print("Full data shape:", df.shape)

    test_size = 723  
    df_test = df.iloc[:test_size]
    df_train = df.iloc[test_size:]

    print(f"Train size: {len(df_train)}  Test size: {len(df_test)}")

    (test_accuracy, train_accuracy, report, model, feature_names, shape,
     train_loss, test_loss) = train_softmax_regression_model(df_train, df_test)

    print("Combined feature matrix shape:", shape)
    print("\nTraining Accuracy: {:.4f}".format(train_accuracy))
    print("Test Accuracy: {:.4f}".format(test_accuracy))
    print("\nTraining Log Loss: {:.4f}".format(train_loss))
    print("Test Log Loss: {:.4f}".format(test_loss))
    print("\nClassification Report:\n", report)

    params_to_save = {
        'coef': model.coef_,
        'intercept': model.intercept_,
        'classes': model.classes_,
        'feature_names': feature_names,
    }

    _, _, q3_word_to_idx, q6_word_to_idx = process_q3q6_data(df_train)
    _, q5_word_to_idx = create_movie_features(df_train)
    _, _, q7_word_to_idx, q8_category_to_idx = process_data(df_train)

    params_to_save['q3_word_to_idx'] = q3_word_to_idx
    params_to_save['q6_word_to_idx'] = q6_word_to_idx
    params_to_save['q5_word_to_idx'] = q5_word_to_idx
    params_to_save['q7_word_to_idx'] = q7_word_to_idx
    params_to_save['q8_category_to_idx'] = q8_category_to_idx

    save_path = "softmax_model_params.npz"
    np.savez(save_path, **params_to_save)
    print(f"\nModel parameters saved to {save_path}")


if __name__ == "__main__":
    main()