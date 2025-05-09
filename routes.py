"""
Flask Routes Module for Behavioral Phishing Detection Application
This module defines all routes and handlers for the Flask web application.
"""

from flask import render_template, Blueprint

# Create Blueprint with a name that matches the URL generation in templates
bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    """Home page route."""
    return render_template('index.html')

@bp.route('/about')
def about():
    """About page route."""
    return render_template('about.html')

@bp.route('/api/docs')
def api_docs():
    """API documentation page."""
    return render_template('api_docs.html')

@bp.route('/demo')
def demo():
    """Demo page with example phishing emails."""
    return render_template('demo.html')