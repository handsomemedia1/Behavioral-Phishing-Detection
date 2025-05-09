"""
Data Loader Module for Phishing Detection

This module handles loading and preparing email datasets for the phishing detection system.
"""

import os
import pandas as pd
import numpy as np
import email
import glob
from tqdm import tqdm

from src.utils.email_parser import parse_email


def load_emails_from_csv(file_path, text_column='text', label_column='is_phishing'):
    """Load emails from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        text_column (str): Column name containing email text
        label_column (str): Column name containing phishing labels
        
    Returns:
        pandas.DataFrame: DataFrame containing emails
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found at {file_path}")
    
    # Load CSV
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {e}")
    
    # Check if required columns exist
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in CSV")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in CSV")
    
    # Rename columns for consistency
    df = df.rename(columns={text_column: 'body', label_column: 'is_phishing'})
    
    # Add email_id if not present
    if 'email_id' not in df.columns:
        df['email_id'] = np.arange(len(df))
    
    return df


def load_emails_from_directory(directory_path, phishing_folder='phishing', legitimate_folder='legitimate'):
    """Load emails from a directory structure.
    
    Args:
        directory_path (str): Path to the directory containing email folders
        phishing_folder (str): Name of the folder containing phishing emails
        legitimate_folder (str): Name of the folder containing legitimate emails
        
    Returns:
        pandas.DataFrame: DataFrame containing emails
    """
    # Check if directory exists
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found at {directory_path}")
    
    # Paths to phishing and legitimate email folders
    phishing_path = os.path.join(directory_path, phishing_folder)
    legitimate_path = os.path.join(directory_path, legitimate_folder)
    
    # Check if folders exist
    if not os.path.exists(phishing_path):
        raise FileNotFoundError(f"Phishing folder not found at {phishing_path}")
    if not os.path.exists(legitimate_path):
        raise FileNotFoundError(f"Legitimate folder not found at {legitimate_path}")
    
    # Lists to store email data
    emails = []
    labels = []
    
    # Load phishing emails
    print("Loading phishing emails...")
    phishing_files = glob.glob(os.path.join(phishing_path, '*'))
    for file_path in tqdm(phishing_files):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                email_content = f.read()
            emails.append(email_content)
            labels.append(1)  # 1 for phishing
        except Exception as e:
            print(f"Error loading phishing email {file_path}: {e}")
    
    # Load legitimate emails
    print("Loading legitimate emails...")
    legitimate_files = glob.glob(os.path.join(legitimate_path, '*'))
    for file_path in tqdm(legitimate_files):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                email_content = f.read()
            emails.append(email_content)
            labels.append(0)  # 0 for legitimate
        except Exception as e:
            print(f"Error loading legitimate email {file_path}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame({
        'email_id': np.arange(len(emails)),
        'raw_content': emails,
        'is_phishing': labels
    })
    
    # Parse emails
    print("Parsing emails...")
    parsed_emails = []
    for email_content in tqdm(df['raw_content']):
        try:
            parsed = parse_email(email_content)
            parsed_emails.append(parsed)
        except Exception as e:
            print(f"Error parsing email: {e}")
            # Add empty parsed email as fallback
            parsed_emails.append({
                'headers': {},
                'subject': "",
                'from': "",
                'to': "",
                'date': "",
                'body': email_content,
                'urls': [],
                'has_attachments': False,
                'is_html': False
            })
    
    # Extract body and other fields from parsed emails
    df['body'] = [parsed['body'] for parsed in parsed_emails]
    df['subject'] = [parsed['subject'] for parsed in parsed_emails]
    df['from'] = [parsed['from'] for parsed in parsed_emails]
    df['to'] = [parsed['to'] for parsed in parsed_emails]
    df['date'] = [parsed['date'] for parsed in parsed_emails]
    df['urls'] = [parsed['urls'] for parsed in parsed_emails]
    df['has_attachments'] = [parsed['has_attachments'] for parsed in parsed_emails]
    df['is_html'] = [parsed['is_html'] for parsed in parsed_emails]
    
    return df


def download_public_dataset(dataset_name='phishtank', output_dir='data/raw'):
    """Download a public phishing email dataset.
    
    Args:
        dataset_name (str): Name of the dataset to download
        output_dir (str): Directory to save the dataset
        
    Returns:
        str: Path to the downloaded dataset
    """
    import requests
    from io import BytesIO
    from zipfile import ZipFile
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Dataset URLs
    dataset_urls = {
        'phishtank': 'https://data.phishtank.com/data/online-valid.csv.zip',
        # Add more datasets as needed
    }
    
    # Check if dataset is supported
    if dataset_name not in dataset_urls:
        raise ValueError(f"Dataset {dataset_name} is not supported. Available datasets: {list(dataset_urls.keys())}")
    
    # Download dataset
    url = dataset_urls[dataset_name]
    print(f"Downloading {dataset_name} dataset from {url}...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
    except Exception as e:
        raise Exception(f"Error downloading dataset: {e}")
    
    # Save dataset
    output_path = os.path.join(output_dir, f"{dataset_name}.csv")
    
    # Extract if it's a zip file
    if url.endswith('.zip'):
        with ZipFile(BytesIO(response.content)) as zip_file:
            csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
            if not csv_files:
                raise ValueError("No CSV files found in the downloaded zip file")
            
            # Extract the first CSV file
            with zip_file.open(csv_files[0]) as csv_file, open(output_path, 'wb') as output_file:
                output_file.write(csv_file.read())
    else:
        with open(output_path, 'wb') as f:
            f.write(response.content)
    
    print(f"Dataset saved to {output_path}")
    return output_path


def split_dataset(df, test_size=0.2, val_size=0.25, random_state=42):
    """Split dataset into train, validation, and test sets.
    
    Args:
        df (pandas.DataFrame): DataFrame containing emails
        test_size (float): Proportion of data to use for testing
        val_size (float): Proportion of training data to use for validation
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    # Split into train+val and test
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['is_phishing']
    )
    
    # Split train+val into train and val
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size, random_state=random_state, stratify=train_val_df['is_phishing']
    )
    
    print(f"Dataset split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    return train_df, val_df, test_df


def save_dataset_splits(train_df, val_df, test_df, output_dir='data/processed'):
    """Save dataset splits to CSV files.
    
    Args:
        train_df (pandas.DataFrame): Training set
        val_df (pandas.DataFrame): Validation set
        test_df (pandas.DataFrame): Test set
        output_dir (str): Directory to save the CSV files
        
    Returns:
        tuple: Paths to the saved CSV files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save dataframes
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Dataset splits saved to {output_dir}")
    
    return train_path, val_path, test_path


if __name__ == "__main__":
    # Example usage
    
    # Load from CSV
    # df = load_emails_from_csv('data/raw/phishing_dataset.csv')
    
    # Or download public dataset
    # dataset_path = download_public_dataset('phishtank')
    # df = load_emails_from_csv(dataset_path)
    
    # Or load from directory
    # df = load_emails_from_directory('data/raw/emails')
    
    # Split dataset
    # train_df, val_df, test_df = split_dataset(df)
    
    # Save splits
    # save_dataset_splits(train_df, val_df, test_df)
    
    print("Data loader module loaded successfully!")