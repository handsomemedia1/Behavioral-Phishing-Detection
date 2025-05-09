"""
Main application file for the Behavioral Phishing Detection system.
This creates a Flask web application with API endpoints for phishing detection.
"""

import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

# Import the blueprint from routes
from routes import bp as main_blueprint
from src.data.preprocessor import clean_email
from src.features.linguistic import LinguisticFeatureExtractor
from src.features.url_analysis import URLFeatureExtractor
from src.features.feature_builder import combine_features
from src.utils.email_parser import parse_email

# Initialize Flask app
app = Flask(__name__)

# Register the blueprint
app.register_blueprint(main_blueprint)

# Load models and scaler
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Set up model and scaler variables to avoid undefined errors
model = None
scaler = None

# Try to load the best model
try:
    MODEL_INFO_PATH = os.path.join(MODELS_DIR, 'best_model_info.json')
    with open(MODEL_INFO_PATH, 'r') as f:
        best_model_info = json.load(f)

    BEST_MODEL_NAME = best_model_info['model_name']

    if BEST_MODEL_NAME == 'Random Forest':
        MODEL_PATH = os.path.join(MODELS_DIR, 'random_forest.pkl')
        model = joblib.load(MODEL_PATH)
    elif BEST_MODEL_NAME == 'SVM':
        MODEL_PATH = os.path.join(MODELS_DIR, 'svm_model.pkl')
        model = joblib.load(MODEL_PATH)
    elif BEST_MODEL_NAME == 'Neural Network':
        import tensorflow as tf
        MODEL_PATH = os.path.join(MODELS_DIR, 'neural_network')
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        print(f"Unknown model type: {BEST_MODEL_NAME}")

    # Load scaler
    SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
    scaler = joblib.load(SCALER_PATH)
    
    print(f"Successfully loaded model: {BEST_MODEL_NAME}")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    print("App will use fallback heuristics instead of ML model")

# Initialize feature extractors
linguistic_extractor = LinguisticFeatureExtractor()
url_extractor = URLFeatureExtractor()

def predict_phishing(email_content):
    """Predict if an email is phishing or legitimate."""
    # Parse and clean email
    email_data = parse_email(email_content)
    clean_text = clean_email(email_data['body'])
    
    # Extract features
    linguistic_features = linguistic_extractor.extract_features(clean_text)
    url_features = url_extractor.extract_features(email_data['urls'])
    
    # Combine features
    features = combine_features(linguistic_features, url_features)
    
    # Try to use the model if available
    try:
        if model is not None and scaler is not None:
            # Convert to array and scale
            feature_names = sorted(features.keys())
            feature_values = np.array([features[name] for name in feature_names]).reshape(1, -1)
            scaled_features = scaler.transform(feature_values)
            
            # Make prediction
            if BEST_MODEL_NAME in ['Random Forest', 'SVM']:
                # Scikit-learn models
                phishing_proba = model.predict_proba(scaled_features)[0, 1]
                is_phishing = model.predict(scaled_features)[0]
            else:
                # Neural network
                phishing_proba = model.predict(scaled_features)[0, 0]
                is_phishing = (phishing_proba > 0.5)
            
            # Extract top contributing features
            if BEST_MODEL_NAME == 'Random Forest':
                # For random forest, we can get feature importances
                importances = model.feature_importances_
                top_indices = np.argsort(importances)[-5:][::-1]
                top_features = [(feature_names[i], importances[i]) for i in top_indices]
            else:
                # For other models, we don't have direct feature importances
                top_features = []
        else:
            raise ValueError("Model or scaler not available")
    except Exception as e:
        print(f"Error using model: {e}")
        # Fallback using heuristics
        suspicious_words = ['urgent', 'verify', 'suspended', 'unauthorized', 'security', 'login']
        suspicious_count = sum(1 for word in suspicious_words if word.lower() in clean_text.lower())
        
        has_suspicious_urls = any(
            '.xyz' in url.lower() or 'verify' in url.lower() or 'secure' in url.lower()
            for url in email_data['urls']
        )
        
        unusual_formatting = clean_text.count('!') > 3
        
        # Calculate phishing probability based on simple heuristics
        phishing_score = (suspicious_count / 6) * 0.5 + (1 if has_suspicious_urls else 0) * 0.3 + (1 if unusual_formatting else 0) * 0.2
        phishing_proba = min(max(phishing_score, 0.1), 0.9)  # Keep between 0.1 and 0.9 for demonstration
        is_phishing = phishing_proba > 0.5
        
        # Demonstrate top features
        top_features = [
            ('ling_urgency_word_count', 0.25),
            ('url_has_suspicious_tld', 0.20),
            ('ling_exclamation_count', 0.18),
            ('ling_threat_word_count', 0.15),
            ('url_has_redirect_url', 0.12)
        ]
    
    # Prepare result
    result = {
        'is_phishing': bool(is_phishing),
        'phishing_probability': float(phishing_proba),
        'confidence': float(max(phishing_proba, 1 - phishing_proba)),
        'top_features': top_features
    }
    
    return result

# Keep the analyze route at the app level
@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze an email via form submission."""
    if 'email' not in request.form:
        return render_template('index.html', error="No email provided")
    
    email_content = request.form['email']
    result = predict_phishing(email_content)
    
    return render_template('result.html', result=result, email=email_content)

# Keep the API endpoint at the app level
@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for email analysis."""
    data = request.json
    
    if not data or 'email_content' not in data:
        return jsonify({
            'error': 'Missing email_content field'
        }), 400
    
    email_content = data['email_content']
    result = predict_phishing(email_content)
    
    return jsonify(result)

# Add a debug route to help diagnose issues
@app.route('/debug-routes')
def debug_routes():
    """Show all registered routes for debugging."""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'rule': str(rule)
        })
    return jsonify(routes)

if __name__ == '__main__':
    # Print debug information
    print(f"Template folder: {app.template_folder}")
    print(f"Templates found: {os.listdir(app.template_folder)}")
    print(f"Registered routes:")
    for rule in app.url_map.iter_rules():
        print(f"  • {rule} → {rule.endpoint}")
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))