import pandas as pd
from sklearn.model_selection import train_test_split
import os


def split_dataset(df,dataset_name,output_folder):

    # Split the dataset based on the 'aspect' column
    df_teacher = df[df['aspect'] == 'the teacher']
    df_course = df[df['aspect'] == 'the course']

    # Create train-test splits for each aspect
    train_teacher, test_teacher = train_test_split(df_teacher, test_size=0.2, random_state=42)
    train_course, test_course = train_test_split(df_course, test_size=0.2, random_state=42)

    # Combine the splits
    train_df = pd.concat([train_teacher, train_course])
    test_df = pd.concat([test_teacher, test_course])

    # Shuffle the datasets
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Select only the required columns
    train_df = train_df[['processed_review', 'aspect', 'label']]
    test_df = test_df[['processed_review', 'aspect', 'label']]

    # Convert labels from 0, 1, 2 to -1, 0, 1
    label_mapping = {0: -1, 1: 0, 2: 1}
    train_df['label'] = train_df['label'].replace(label_mapping)
    test_df['label'] = test_df['label'].replace(label_mapping)

    # Save the resulting datasets to .seg files without headers
    output_train_file = os.path.join(output_folder, f'train_{dataset_name}.seg')
    output_test_file = os.path.join(output_folder, f'test_{dataset_name}.seg')

    train_df.to_csv(output_train_file, sep='\n', header=False, index=False)
    test_df.to_csv(output_test_file, sep='\n', header=False, index=False)

    print("Datasets split and saved successfully in .seg format.")