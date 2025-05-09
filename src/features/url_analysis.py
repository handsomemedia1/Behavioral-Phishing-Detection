"""
URL Feature Extraction Module for Phishing Detection

This module extracts features from URLs found in emails to detect phishing attempts.
"""

import re
import numpy as np
from urllib.parse import urlparse, parse_qs
import tldextract

class URLFeatureExtractor:
    """Extract features from URLs for phishing detection."""
    
    def __init__(self):
        """Initialize the URL feature extractor."""
        # Suspicious TLDs often used in phishing
        self.suspicious_tlds = {
            'tk', 'xyz', 'ml', 'ga', 'cf', 'gq', 'top', 'date', 
            'win', 'review', 'stream', 'online', 'site', 'tech', 'fit'
        }
        
        # Common legitimate domains not typically used in phishing
        self.common_domains = {
            'google.com', 'microsoft.com', 'apple.com', 'amazon.com', 'facebook.com', 
            'github.com', 'linkedin.com', 'twitter.com', 'instagram.com', 'youtube.com',
            'yahoo.com', 'netflix.com', 'wikipedia.org', 'dropbox.com', 'wordpress.com',
            'outlook.com', 'office.com', 'live.com', 'adobe.com', 'paypal.com',
            'shopify.com', 'airbnb.com', 'uber.com', 'lyft.com', 'ebay.com',
            'zoom.us', 'slack.com', 'spotify.com', 'pinterest.com', 'reddit.com'
        }
    
    def extract_features(self, urls):
        """Extract features from a list of URLs.
        
        Args:
            urls (list): List of URL strings
            
        Returns:
            dict: Dictionary of extracted features
        """
        if not urls:
            # Return default features for no URLs
            return self._get_default_features()
        
        # Ensure urls is a list
        if isinstance(urls, str):
            urls = [urls]
        
        # Analyze each URL
        url_features = [self._analyze_single_url(url) for url in urls]
        
        # Aggregate features across all URLs
        aggregated_features = self._aggregate_url_features(url_features)
        
        return aggregated_features
    
    def _get_default_features(self):
        """Return default features for when no URLs are present."""
        return {
            'url_count': 0,
            'avg_url_length': 0,
            'max_url_length': 0,
            'has_https_url': 0,
            'has_ip_url': 0,
            'has_suspicious_tld': 0,
            'has_suspicious_subdomain': 0,
            'has_multiple_subdomains': 0,
            'max_subdomain_count': 0,
            'has_url_with_port': 0,
            'has_url_with_query_params': 0,
            'max_query_params_count': 0,
            'has_url_with_fragment': 0,
            'has_numeric_domain': 0,
            'has_domain_with_dash': 0,
            'has_common_domain': 0,
            'has_uncommon_port': 0,
            'url_length_disparity': 0,
            'has_redirect_url': 0,
            'has_url_shortener': 0,
            'domain_age_disparity': 0
        }
    
    def _analyze_single_url(self, url):
        """Analyze a single URL for phishing indicators.
        
        Args:
            url (str): URL string
            
        Returns:
            dict: Features extracted from the URL
        """
        # Ensure URL has a scheme
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        # Parse URL components
        parsed_url = urlparse(url)
        extracted = tldextract.extract(url)
        
        # Get domain and subdomain info
        domain = extracted.domain + '.' + extracted.suffix if extracted.suffix else extracted.domain
        subdomain = extracted.subdomain
        
        # Extract features
        features = {
            'url_length': len(url),
            'is_https': 1 if parsed_url.scheme == 'https' else 0,
            'has_ip': 1 if self._is_ip_address(parsed_url.netloc) else 0,
            'has_suspicious_tld': 1 if extracted.suffix in self.suspicious_tlds else 0,
            'has_suspicious_subdomain': 1 if self._has_suspicious_subdomain(subdomain) else 0,
            'subdomain_count': subdomain.count('.') + 1 if subdomain else 0,
            'has_port': 1 if parsed_url.port is not None else 0,
            'has_query_params': 1 if parsed_url.query else 0,
            'query_params_count': len(parse_qs(parsed_url.query)),
            'has_fragment': 1 if parsed_url.fragment else 0,
            'has_numeric_domain': 1 if re.search(r'\d', domain) else 0,
            'has_dash_in_domain': 1 if '-' in domain else 0,
            'is_common_domain': 1 if domain in self.common_domains else 0,
            'has_uncommon_port': 1 if self._has_uncommon_port(parsed_url.port) else 0,
            'is_url_shortener': 1 if self._is_url_shortener(domain) else 0,
            'has_redirect': 1 if self._has_redirect_pattern(url) else 0
        }
        
        return features
    
    def _aggregate_url_features(self, url_features_list):
        """Aggregate features from multiple URLs.
        
        Args:
            url_features_list (list): List of URL feature dictionaries
            
        Returns:
            dict: Aggregated features
        """
        if not url_features_list:
            return self._get_default_features()
        
        # Number of URLs
        url_count = len(url_features_list)
        
        # URL lengths
        url_lengths = [features['url_length'] for features in url_features_list]
        avg_url_length = np.mean(url_lengths) if url_lengths else 0
        max_url_length = max(url_lengths) if url_lengths else 0
        url_length_disparity = np.std(url_lengths) if len(url_lengths) > 1 else 0
        
        # Suspicious indicators
        has_https_url = any(features['is_https'] for features in url_features_list)
        has_ip_url = any(features['has_ip'] for features in url_features_list)
        has_suspicious_tld = any(features['has_suspicious_tld'] for features in url_features_list)
        has_suspicious_subdomain = any(features['has_suspicious_subdomain'] for features in url_features_list)
        has_multiple_subdomains = any(features['subdomain_count'] > 1 for features in url_features_list)
        max_subdomain_count = max(features['subdomain_count'] for features in url_features_list) if url_features_list else 0
        
        # URL components
        has_url_with_port = any(features['has_port'] for features in url_features_list)
        has_url_with_query_params = any(features['has_query_params'] for features in url_features_list)
        max_query_params_count = max(features['query_params_count'] for features in url_features_list) if url_features_list else 0
        has_url_with_fragment = any(features['has_fragment'] for features in url_features_list)
        
        # Domain characteristics
        has_numeric_domain = any(features['has_numeric_domain'] for features in url_features_list)
        has_domain_with_dash = any(features['has_dash_in_domain'] for features in url_features_list)
        has_common_domain = any(features['is_common_domain'] for features in url_features_list)
        has_uncommon_port = any(features['has_uncommon_port'] for features in url_features_list)
        
        # Redirection and shortening
        has_redirect_url = any(features['has_redirect'] for features in url_features_list)
        has_url_shortener = any(features['is_url_shortener'] for features in url_features_list)
        
        # Domain variation (indicator of registered similar domains)
        domain_set = set()
        for url_features in url_features_list:
            parsed = urlparse(url)
            domain_set.add(parsed.netloc)
        domain_age_disparity = 1 if len(domain_set) > 1 else 0
        
        # Combine all features
        return {
            'url_count': url_count,
            'avg_url_length': avg_url_length,
            'max_url_length': max_url_length,
            'has_https_url': int(has_https_url),
            'has_ip_url': int(has_ip_url),
            'has_suspicious_tld': int(has_suspicious_tld),
            'has_suspicious_subdomain': int(has_suspicious_subdomain),
            'has_multiple_subdomains': int(has_multiple_subdomains),
            'max_subdomain_count': max_subdomain_count,
            'has_url_with_port': int(has_url_with_port),
            'has_url_with_query_params': int(has_url_with_query_params),
            'max_query_params_count': max_query_params_count,
            'has_url_with_fragment': int(has_url_with_fragment),
            'has_numeric_domain': int(has_numeric_domain),
            'has_domain_with_dash': int(has_domain_with_dash),
            'has_common_domain': int(has_common_domain),
            'has_uncommon_port': int(has_uncommon_port),
            'url_length_disparity': url_length_disparity,
            'has_redirect_url': int(has_redirect_url),
            'has_url_shortener': int(has_url_shortener),
            'domain_age_disparity': domain_age_disparity
        }
    
    def _is_ip_address(self, domain):
        """Check if a domain is an IP address.
        
        Args:
            domain (str): Domain string
            
        Returns:
            bool: True if domain is an IP address
        """
        # IPv4 pattern
        ipv4_pattern = r'^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})
        match = re.match(ipv4_pattern, domain)
        
        if match:
            # Check if each octet is valid (0-255)
            return all(0 <= int(octet) <= 255 for octet in match.groups())
        
        return False
    
    def _has_suspicious_subdomain(self, subdomain):
        """Check if a subdomain appears suspicious.
        
        Args:
            subdomain (str): Subdomain string
            
        Returns:
            bool: True if subdomain appears suspicious
        """
        if not subdomain:
            return False
        
        # Check for excessive length
        if len(subdomain) > 30:
            return True
        
        # Check for excessive dashes
        if subdomain.count('-') > 3:
            return True
        
        # Check for suspicious terms in subdomain
        suspicious_terms = [
            'secure', 'login', 'signin', 'banking', 'account', 'verify',
            'update', 'confirm', 'authenticate', 'validation', 'billing'
        ]
        
        for term in suspicious_terms:
            if term in subdomain.lower():
                return True
        
        return False
    
    def _has_uncommon_port(self, port):
        """Check if a port is uncommon (not standard HTTP/HTTPS).
        
        Args:
            port (int): Port number
            
        Returns:
            bool: True if port is uncommon
        """
        if port is None:
            return False
        
        # Standard HTTP/HTTPS ports
        common_ports = {80, 443, 8080, 8443}
        
        return port not in common_ports
    
    def _is_url_shortener(self, domain):
        """Check if a domain is a URL shortener service.
        
        Args:
            domain (str): Domain string
            
        Returns:
            bool: True if domain is a URL shortener
        """
        # Common URL shortener domains
        shortener_domains = {
            'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'tiny.cc',
            'is.gd', 'cli.gs', 'pic.gd', 'DwarfURL.com', 'ow.ly',
            'yfrog.com', 'migre.me', 'ff.im', 'tiny.pl', 'url4.eu',
            'tr.im', 'twit.ac', 'su.pr', 'twurl.nl', 'snipurl.com',
            'short.to', 'BudURL.com', 'ping.fm', 'post.ly', 'Just.as',
            'bkite.com', 'snipr.com', 'fic.kr', 'loopt.us', 'doiop.com',
            'twitthis.com', 'htxt.it', 'AltURL.com', 'RedirX.com', 'DigBig.com',
            'short.ie', 'u.mavrev.com', 'kl.am', 'wp.me', 'u.nu', 'rubyurl.com',
            'om.ly', 'linkbee.com', 'Yep.it', 'posted.at', 'xrl.us', 'metamark.net',
            'sn.im', 'hurl.ws', 'eepurl.com', 'idek.net', 'urlpire.com', 'chilp.it',
            'moourl.com', 'snurl.com', 'xr.com', 'lin.cr', 'shortlinks.co.uk',
            'go2l.ink', 'x.co', 'prettylinkpro.com', 'viralurl.com', 'qr.net',
            'cutt.ly', 'shorturl.at'
        }
        
        return domain.lower() in shortener_domains
    
    def _has_redirect_pattern(self, url):
        """Check if a URL contains redirect patterns.
        
        Args:
            url (str): URL string
            
        Returns:
            bool: True if URL contains redirect patterns
        """
        # Common redirect patterns in URLs
        redirect_patterns = [
            'redirect', 'redir', 'url=', 'link=', 'goto=', 'go?url=',
            'return=', 'returl=', 'urlback=', 'location=', 'jump=',
            'cgi-bin', 'forward=', 'target='
        ]
        
        url_lower = url.lower()
        return any(pattern in url_lower for pattern in redirect_patterns)


if __name__ == "__main__":
    # Example usage
    extractor = URLFeatureExtractor()
    
    # Example URLs
    urls = [
        'https://legitimate-bank.com/login',
        'http://phishing-site.xyz/login?redirect=https://legitimate-bank.com',
        'http://192.168.1.1/login',
        'https://bit.ly/2short',
        'https://bank-secure-login-verification-update.tk/account'
    ]
    
    # Extract features
    features = extractor.extract_features(urls)
    
    # Print features
    for feature, value in features.items():
        print(f"{feature}: {value}")