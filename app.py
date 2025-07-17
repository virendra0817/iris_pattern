from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Global variables for the model
clf = None
label_encoder = None

def load_model():
    """Load and train the iris classification model"""
    global clf, label_encoder
    
    # Create sample iris data
    print("Using sample iris data...")
    data = {
        'sepal_length': [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9, 7.0, 6.4, 6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2, 6.3, 3.3, 5.8, 7.1, 6.3, 5.8, 7.6, 4.9, 7.3, 6.7],
        'sepal_width': [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1, 3.2, 3.2, 3.1, 2.3, 2.8, 2.8, 3.3, 2.4, 2.9, 2.7, 3.3, 3.0, 2.7, 3.0, 2.9, 2.7, 3.0, 2.5, 2.9, 2.5],
        'petal_length': [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5, 4.7, 4.5, 4.9, 4.0, 4.6, 4.5, 4.7, 3.3, 4.6, 3.9, 6.0, 5.0, 5.1, 5.9, 5.6, 5.1, 6.6, 4.5, 6.3, 5.8],
        'petal_width': [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 1.4, 1.5, 1.5, 1.3, 1.5, 1.3, 1.6, 1.0, 1.3, 1.4, 2.5, 1.9, 1.9, 2.1, 1.8, 1.9, 2.1, 1.7, 1.8, 1.8],
        'species': ['setosa']*10 + ['versicolor']*10 + ['virginica']*10
    }
    df = pd.DataFrame(data)
    
    # Prepare features and target
    features = df.iloc[:, 0:4]
    target = df.iloc[:, -1]
    
    # Encode target labels
    label_encoder = LabelEncoder()
    target_encoded = label_encoder.fit_transform(target)
    
    # Train the model
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(features, target_encoded)
    
    print("Model loaded and trained successfully!")

@app.route('/')
def home():
    """Main page with the classification form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        
        # Validate inputs
        if not (0 < sepal_length < 10 and 0 < sepal_width < 10 and 
                0 < petal_length < 10 and 0 < petal_width < 10):
            return jsonify({'error': 'Please enter valid measurements between 0 and 10 cm'})
        
        # Make prediction
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = clf.predict(features)[0]
        probability = clf.predict_proba(features)[0]
        
        # Get species name
        species_name = label_encoder.inverse_transform([prediction])[0]
        confidence = max(probability) * 100
        
        # Format species name nicely
        species_display = f"Iris {species_name.title()}"
        
        return jsonify({
            'species': species_display,
            'confidence': round(confidence, 1),
            'success': True
        })
        
    except ValueError:
        return jsonify({'error': 'Please enter valid numbers for all measurements'})
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'})

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)