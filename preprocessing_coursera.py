
import argparse
import pandas as pd
from tqdm import tqdm
from preprocessing_dataset import preprocess_dataset
from split_dataset import split_dataset

# Initialize progress bar
tqdm.pandas()

# Define constants
DEFAULT_DATASET_URL = './datasets/coursera/Coursera_dataset.csv'
DEFAULT_OUTPUT_FOLDER = './datasets/coursera/'
PROCESSED_COLUMNS = [
    'neg_false_keep_all',
    'neg_false_del_all',
    'neg_false_del_except_neg',
    'neg_true_keep_all',
    'neg_true_del_all',
    'neg_true_del_except_neg',
    'aspect', 'label'
]
def main():
    """Main function to preprocess and split the dataset."""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Preprocess and split dataset.')
    parser.add_argument('--dataset_url', default=DEFAULT_DATASET_URL, type=str, help='Path to dataset file (CSV)')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str, help='Output folder for processed files')
    args = parser.parse_args()

    # Load dataset
    df_data = load_and_clean_data(args.dataset_url)

    # Preprocess dataset with different configurations
    df_data = preprocess_reviews(df_data)

    # Extract file name (or use a default)
    file_name = "preprocessed_coursera"

    # Split the dataset
    split_dataset(df_data, file_name, args.output_folder, PROCESSED_COLUMNS)


def load_and_clean_data(dataset_url: str) -> pd.DataFrame:
    """Load dataset from CSV, clean by removing NaN values."""
    chunks = pd.read_csv(dataset_url, chunksize=1000)
    df = pd.concat(chunks)
    return df.dropna()


def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing to reviews with various negation and stopword configurations."""
    configurations = [
        (False, "none", 'processed_neg_false_keep_all'),
        (False, "all", 'processed_neg_false_del_all'),
        (False, "all_except_neg", 'processed_neg_false_del_except_neg'),
        (True, "none", 'processed_neg_true_keep_all'),
        (True, "all", 'processed_neg_true_del_all'),
        (True, "all_except_neg", 'processed_neg_true_del_except_neg'),
    ]

    for negation, stopword, column in configurations:
        df[column] = df['review'].progress_apply(preprocess_dataset, 
                                                 negation_handling_on=negation, 
                                                 stop_word_delete=stopword)
    return df


if __name__ == "__main__":
    main()
