"""
Configuration settings for the Behavioral Phishing Detection system.
"""

import os

# Base directory of the project
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Flask configurations
class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard-to-guess-string'
    DEBUG = False
    TESTING = False
    
    # Model settings
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    DEFAULT_MODEL = 'random_forest'  # Default model to use if best model not found
    
    # Data directory
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    
    # API settings
    API_RATE_LIMIT = 100  # Requests per hour


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration."""
    # Production settings
    DEBUG = False
    
    # Security settings
    SESSION_COOKIE_SECURE = True
    REMEMBER_COOKIE_SECURE = True
    
    # Set this to your domain in production
    SERVER_NAME = os.environ.get('SERVER_NAME')


# Configuration dictionary
config_dict = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}


def get_config():
    """Get the configuration based on the environment.
    
    Returns:
        Config: Configuration class
    """
    config_name = os.environ.get('FLASK_CONFIG', 'default')
    return config_dict.get(config_name, config_dict['default'])