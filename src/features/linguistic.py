"""
Linguistic Feature Extraction Module for Phishing Detection

This module extracts linguistic features from email text that can be used
to detect phishing attempts based on writing patterns.
"""

import re
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from textstat import flesch_reading_ease, syllable_count
import spacy

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Load NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Common phishing language patterns
URGENCY_WORDS = [
    'urgent', 'immediately', 'alert', 'warning', 'attention', 'important',
    'critical', 'verify', 'suspended', 'limited', 'expired', 'deadline'
]

THREAT_WORDS = [
    'suspended', 'terminated', 'deleted', 'blocked', 'unauthorized',
    'suspicious', 'locked', 'limited', 'restriction', 'security', 
    'fraud', 'illegal', 'violation'
]

REWARD_WORDS = [
    'congratulations', 'selected', 'winner', 'prize', 'reward', 'exclusive',
    'free', 'bonus', 'gift', 'discount', 'offer', 'limited time'
]

class LinguisticFeatureExtractor:
    """Extract linguistic features from email text for phishing detection."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.stop_words = set(stopwords.words('english'))
    
    def extract_features(self, email_text):
        """Extract all linguistic features from the email text.
        
        Args:
            email_text (str): The preprocessed email text
            
        Returns:
            dict: Dictionary of extracted features
        """
        if not email_text or not isinstance(email_text, str):
            raise ValueError("Email text must be a non-empty string")
            
        # Create SpaCy document
        doc = nlp(email_text)
        
        # Extract all features
        features = {}
        features.update(self._extract_basic_stats(email_text))
        features.update(self._extract_readability_metrics(email_text))
        features.update(self._extract_urgency_features(email_text))
        features.update(self._extract_grammar_features(doc))
        features.update(self._extract_sentiment_features(doc))
        
        return features
    
    def _extract_basic_stats(self, text):
        """Extract basic statistical features from text."""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        words_without_stopwords = [w for w in words if w.lower() not in self.stop_words]
        
        return {
            'num_sentences': len(sentences),
            'num_words': len(words),
            'num_chars': len(text),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'avg_sentence_length': np.mean([len(word_tokenize(s)) for s in sentences]) if sentences else 0,
            'lexical_diversity': len(set(words)) / len(words) if words else 0,
            'non_stopwords_ratio': len(words_without_stopwords) / len(words) if words else 0
        }
    
    def _extract_readability_metrics(self, text):
        """Extract readability metrics from text."""
        try:
            readability_score = flesch_reading_ease(text)
        except:
            readability_score = 50  # Default middle value
            
        # Calculate syllables per word
        words = word_tokenize(text)
        syllables = [syllable_count(word) for word in words]
        avg_syllables = np.mean(syllables) if syllables else 0
        
        return {
            'readability_score': readability_score,
            'avg_syllables_per_word': avg_syllables
        }
    
    def _extract_urgency_features(self, text):
        """Extract features related to urgency and manipulation."""
        text_lower = text.lower()
        
        # Count urgency indicators
        urgency_count = sum(1 for word in URGENCY_WORDS if word in text_lower)
        threat_count = sum(1 for word in THREAT_WORDS if word in text_lower)
        reward_count = sum(1 for word in REWARD_WORDS if word in text_lower)
        
        # Count exclamation marks and all caps words
        exclamation_count = text.count('!')
        all_caps_count = len(re.findall(r'\b[A-Z]{2,}\b', text))
        
        # Check for time pressure phrases
        time_pressure = any(phrase in text_lower for phrase in [
            'act now', 'limited time', 'expires', 'deadline', 'today only',
            'within 24 hours', 'immediate', 'urgent'
        ])
        
        return {
            'urgency_word_count': urgency_count,
            'threat_word_count': threat_count,
            'reward_word_count': reward_count,
            'exclamation_count': exclamation_count,
            'all_caps_count': all_caps_count,
            'has_time_pressure': int(time_pressure)
        }
    
    def _extract_grammar_features(self, doc):
        """Extract grammar and syntax related features using SpaCy."""
        # POS tag distribution
        pos_counts = {}
        for token in doc:
            pos = token.pos_
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        
        # Normalize by total tokens
        total_tokens = len(doc)
        for pos in pos_counts:
            pos_counts[pos] = pos_counts[pos] / total_tokens if total_tokens else 0
        
        # Extract specific POS ratios that might be relevant
        features = {
            'noun_ratio': pos_counts.get('NOUN', 0),
            'verb_ratio': pos_counts.get('VERB', 0),
            'adj_ratio': pos_counts.get('ADJ', 0),
            'adv_ratio': pos_counts.get('ADV', 0),
            'pron_ratio': pos_counts.get('PRON', 0)
        }
        
        # Grammar errors (simplified - just looking for common patterns)
        # In a real implementation, you would use a more sophisticated grammar checker
        grammar_errors = 0
        for sentence in doc.sents:
            # Very simplified check - just an example
            if len(sentence) > 3:  # Only check non-trivial sentences
                has_subject = False
                has_verb = False
                for token in sentence:
                    if token.dep_ in ('nsubj', 'nsubjpass'):
                        has_subject = True
                    if token.pos_ == 'VERB':
                        has_verb = True
                if not (has_subject and has_verb):
                    grammar_errors += 1
        
        features['grammar_error_ratio'] = grammar_errors / len(list(doc.sents)) if doc.sents else 0
        
        return features
    
    def _extract_sentiment_features(self, doc):
        """Extract sentiment-related features using SpaCy."""
        # This is a simplified sentiment analysis
        # In a real implementation, you might use a more sophisticated sentiment analyzer
        
        # Count positive and negative words using SpaCy's lexical attributes
        positive_count = 0
        negative_count = 0
        
        for token in doc:
            if not token.is_stop and not token.is_punct:
                # This is simplified - in reality you'd use a proper sentiment lexicon
                if token._.has_attribute('is_positive'):
                    positive_count += token._.is_positive
                if token._.has_attribute('is_negative'):
                    negative_count += token._.is_negative
        
        # If custom attributes aren't available, use a simple list approach
        # (In a real implementation, you'd use a proper sentiment lexicon)
        if positive_count == 0 and negative_count == 0:
            simple_pos_words = ['good', 'great', 'excellent', 'thank', 'thanks', 'appreciate']
            simple_neg_words = ['bad', 'wrong', 'error', 'problem', 'issue', 'sorry']
            
            positive_count = sum(1 for token in doc if token.lemma_.lower() in simple_pos_words)
            negative_count = sum(1 for token in doc if token.lemma_.lower() in simple_neg_words)
        
        total_tokens = len([t for t in doc if not t.is_stop and not t.is_punct])
        
        return {
            'positive_word_ratio': positive_count / total_tokens if total_tokens else 0,
            'negative_word_ratio': negative_count / total_tokens if total_tokens else 0,
            'sentiment_polarity': (positive_count - negative_count) / (positive_count + negative_count) if (positive_count + negative_count) > 0 else 0
        }


def extract_linguistic_features_from_emails(emails_df, text_column='clean_text'):
    """Extract linguistic features from a dataframe of emails.
    
    Args:
        emails_df (pandas.DataFrame): DataFrame containing emails
        text_column (str): Column name containing the email text
        
    Returns:
        pandas.DataFrame: DataFrame with extracted features
    """
    extractor = LinguisticFeatureExtractor()
    features_list = []
    
    for idx, row in emails_df.iterrows():
        email_text = row[text_column]
        try:
            features = extractor.extract_features(email_text)
            features['email_id'] = idx
            features_list.append(features)
        except Exception as e:
            print(f"Error processing email {idx}: {e}")
            continue
    
    features_df = pd.DataFrame(features_list)
    return features_df

if __name__ == "__main__":
    # Example usage
    sample_text = """URGENT: Your account has been COMPROMISED! 
    You must verify your information IMMEDIATELY or your account will be SUSPENDED within 24 HOURS! 
    Click here to secure your account: http://secure-verify-account.com"""
    
    extractor = LinguisticFeatureExtractor()
    features = extractor.extract_features(sample_text)
    
    for feature, value in features.items():
        print(f"{feature}: {value}")