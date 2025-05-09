"""
Email Preprocessor for Phishing Detection

This module handles the preprocessing of email content for feature extraction.
It includes functions for cleaning text and standardizing email content.
"""

import re
import html
import string
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Get stopwords
STOPWORDS = set(stopwords.words('english'))

def clean_html(text):
    """Remove HTML tags from text.
    
    Args:
        text (str): HTML text
        
    Returns:
        str: Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Use BeautifulSoup to remove HTML tags
    soup = BeautifulSoup(text, 'html.parser')
    
    # Get text content
    clean_text = soup.get_text(separator=' ', strip=True)
    
    # Decode HTML entities
    clean_text = html.unescape(clean_text)
    
    return clean_text

def remove_urls(text):
    """Remove URLs from text.
    
    Args:
        text (str): Text containing URLs
        
    Returns:
        str: Text with URLs removed
    """
    if not text or not isinstance(text, str):
        return ""
    
    # URL pattern
    url_pattern = r'https?://\S+|www\.\S+'
    
    # Replace URLs with a placeholder
    clean_text = re.sub(url_pattern, ' [URL] ', text)
    
    return clean_text

def remove_email_addresses(text):
    """Remove email addresses from text.
    
    Args:
        text (str): Text containing email addresses
        
    Returns:
        str: Text with email addresses removed
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Email pattern
    email_pattern = r'\S+@\S+\.\S+'
    
    # Replace email addresses with a placeholder
    clean_text = re.sub(email_pattern, ' [EMAIL] ', text)
    
    return clean_text

def remove_special_characters(text):
    """Remove special characters from text.
    
    Args:
        text (str): Text with special characters
        
    Returns:
        str: Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    clean_text = text.translate(translator)
    
    # Remove non-ASCII characters
    clean_text = unicodedata.normalize('NFKD', clean_text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    return clean_text

def remove_extra_whitespace(text):
    """Remove extra whitespace from text.
    
    Args:
        text (str): Text with extra whitespace
        
    Returns:
        str: Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Replace multiple whitespace with a single space
    clean_text = re.sub(r'\s+', ' ', text)
    
    # Strip leading and trailing whitespace
    clean_text = clean_text.strip()
    
    return clean_text

def normalize_text(text):
    """Normalize text by converting to lowercase.
    
    Args:
        text (str): Text to normalize
        
    Returns:
        str: Normalized text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    return text.lower()

def remove_stopwords(text):
    """Remove stopwords from text.
    
    Args:
        text (str): Text with stopwords
        
    Returns:
        str: Text without stopwords
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Tokenize
    words = word_tokenize(text)
    
    # Remove stopwords
    filtered_words = [word for word in words if word.lower() not in STOPWORDS]
    
    # Join back into a string
    return ' '.join(filtered_words)

def clean_email(text, remove_stops=False):
    """Clean and preprocess email text.
    
    Args:
        text (str): Raw email text
        remove_stops (bool): Whether to remove stopwords
        
    Returns:
        str: Preprocessed email text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Apply preprocessing steps
    clean_text = text
    clean_text = clean_html(clean_text)
    clean_text = remove_urls(clean_text)
    clean_text = remove_email_addresses(clean_text)
    clean_text = normalize_text(clean_text)
    
    # Only remove stopwords if explicitly requested
    # We often want to keep them for linguistic analysis
    if remove_stops:
        clean_text = remove_stopwords(clean_text)
    
    # These steps are applied last to ensure clean formatting
    clean_text = remove_special_characters(clean_text)
    clean_text = remove_extra_whitespace(clean_text)
    
    return clean_text

def preprocess_email_dataset(emails, content_column='body'):
    """Preprocess a dataset of emails.
    
    Args:
        emails (pandas.DataFrame): DataFrame containing emails
        content_column (str): Column name containing email content
        
    Returns:
        pandas.DataFrame: DataFrame with added clean_text column
    """
    # Create a copy to avoid modifying the original
    emails_df = emails.copy()
    
    # Add a column with cleaned text
    emails_df['clean_text'] = emails_df[content_column].apply(clean_email)
    
    return emails_df


if __name__ == "__main__":
    # Example usage
    sample_email = """
    <html>
    <body>
    <p>Dear Customer,</p>
    <p>We have noticed suspicious activity on your account. Please verify your information by clicking on this link: https://malicious-site.com</p>
    <p>If you don't respond within 24 hours, your account will be <b>SUSPENDED</b>!</p>
    <p>Contact us at support@fake-bank.com for assistance.</p>
    </body>
    </html>
    """
    
    clean_text = clean_email(sample_email)
    print("Original text:")
    print(sample_email)
    print("\nCleaned text:")
    print(clean_text)