"""
Feature Builder Module for Phishing Detection

This module combines features from different extractors into a single feature set.
"""

import pandas as pd


def combine_features(linguistic_features, url_features, metadata_features=None):
    """Combine features from different extractors.
    
    Args:
        linguistic_features (dict): Features from linguistic analysis
        url_features (dict): Features from URL analysis
        metadata_features (dict, optional): Features from email metadata
        
    Returns:
        dict: Combined features
    """
    # Initialize combined features
    combined_features = {}
    
    # Add linguistic features
    for key, value in linguistic_features.items():
        combined_features[f'ling_{key}'] = value
    
    # Add URL features
    for key, value in url_features.items():
        combined_features[f'url_{key}'] = value
    
    # Add metadata features if provided
    if metadata_features:
        for key, value in metadata_features.items():
            combined_features[f'meta_{key}'] = value
    
    return combined_features


def build_feature_matrix(emails_df, linguistic_features_df, url_features_df, metadata_features_df=None):
    """Build a feature matrix from different feature sets.
    
    Args:
        emails_df (pandas.DataFrame): DataFrame with email data
        linguistic_features_df (pandas.DataFrame): DataFrame with linguistic features
        url_features_df (pandas.DataFrame): DataFrame with URL features
        metadata_features_df (pandas.DataFrame, optional): DataFrame with metadata features
        
    Returns:
        pandas.DataFrame: Combined feature matrix
    """
    # Make sure all features have an email_id column for joining
    if 'email_id' not in linguistic_features_df.columns:
        raise ValueError("linguistic_features_df must have an email_id column")
    if 'email_id' not in url_features_df.columns:
        raise ValueError("url_features_df must have an email_id column")
    if metadata_features_df is not None and 'email_id' not in metadata_features_df.columns:
        raise ValueError("metadata_features_df must have an email_id column")
    
    # Merge features
    features_df = pd.merge(
        linguistic_features_df, 
        url_features_df, 
        on='email_id', 
        suffixes=('_ling', '_url')
    )
    
    # Add metadata features if provided
    if metadata_features_df is not None:
        features_df = pd.merge(
            features_df,
            metadata_features_df,
            on='email_id',
            suffixes=('', '_meta')
        )
    
    # Add target variable from emails_df if available
    if 'is_phishing' in emails_df.columns:
        features_df = pd.merge(
            features_df,
            emails_df[['email_id', 'is_phishing']],
            on='email_id'
        )
    
    return features_df


def select_features(features_df, feature_importance=None, threshold=0.01):
    """Select important features based on feature importance.
    
    Args:
        features_df (pandas.DataFrame): DataFrame with all features
        feature_importance (dict, optional): Dict mapping feature names to importance scores
        threshold (float, optional): Minimum importance score for a feature to be selected
        
    Returns:
        pandas.DataFrame: DataFrame with selected features
    """
    # If no feature importance provided, return all features
    if feature_importance is None:
        return features_df
    
    # Get columns to keep (always keep email_id and is_phishing if present)
    columns_to_keep = ['email_id']
    if 'is_phishing' in features_df.columns:
        columns_to_keep.append('is_phishing')
    
    # Add important features
    for feature, importance in feature_importance.items():
        if importance >= threshold and feature in features_df.columns:
            columns_to_keep.append(feature)
    
    # Select features
    selected_features_df = features_df[columns_to_keep]
    
    return selected_features_df


if __name__ == "__main__":
    # Example usage
    linguistic_features = {
        'num_sentences': 10,
        'avg_word_length': 5.2,
        'readability_score': 65.4,
        'urgency_word_count': 3
    }
    
    url_features = {
        'url_count': 2,
        'has_ip_url': 1,
        'has_suspicious_tld': 1
    }
    
    # Combine features
    combined_features = combine_features(linguistic_features, url_features)
    
    # Print combined features
    for feature, value in combined_features.items():
        print(f"{feature}: {value}")