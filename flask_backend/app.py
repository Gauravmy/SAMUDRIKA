"""
Flask main application: ML training, prediction, and image enhancement.

Run:
    flask run
    # or explicitly
    python -m flask --app app run

Test with curl or Postman, or run:
    python test_api.py

Example curl commands:
    # Health check
    curl http://127.0.0.1:5000/health
    
    # Train linear regression (no file, uses dummy data)
    curl -X POST http://127.0.0.1:5000/train/linear
    
    # Train with CSV file
    curl -X POST http://127.0.0.1:5000/train/linear -F "file=@data.csv"
    
    # Train multiple linear
    curl -X POST http://127.0.0.1:5000/train/multiple
    
    # Train binary logistic
    curl -X POST http://127.0.0.1:5000/train/logistic/binary
    
    # Train multiclass logistic
    curl -X POST http://127.0.0.1:5000/train/logistic/multiclass
    
    # Train dummy
    curl -X POST http://127.0.0.1:5000/train/dummy
    
    # Predict
    curl -X POST http://127.0.0.1:5000/predict \
      -H "Content-Type: application/json" \
      -d '{"model_name": "linear", "input": [1.5]}'
    
    # Enhance image
    curl -X POST http://127.0.0.1:5000/enhance/image \
      -F "file=@image.jpg"
"""
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import io
import sys
import os
from importlib import import_module  # type: ignore[import-not-found]

# Add flask_backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules dynamically to avoid Pylance errors
# (These work at runtime even though Pylance can't resolve them statically)
models_store_module = import_module('models_store')  # type: ignore[assignment]
save_model = models_store_module.save_model  # type: ignore[attr-defined]
get_model = models_store_module.get_model  # type: ignore[attr-defined]
list_models = models_store_module.list_models  # type: ignore[attr-defined]

ml_utils = import_module('utils.ml_utils')  # type: ignore[assignment]
train_linear_regression = ml_utils.train_linear_regression  # type: ignore[attr-defined]
train_multiple_linear_regression = ml_utils.train_multiple_linear_regression  # type: ignore[attr-defined]
train_logistic_binary = ml_utils.train_logistic_binary  # type: ignore[attr-defined]
train_logistic_multiclass = ml_utils.train_logistic_multiclass  # type: ignore[attr-defined]
train_dummy_regression = ml_utils.train_dummy_regression  # type: ignore[attr-defined]
predict = ml_utils.predict  # type: ignore[attr-defined]
load_csv_from_buffer = ml_utils.load_csv_from_buffer  # type: ignore[attr-defined]

image_utils_module = import_module('utils.image_utils')  # type: ignore[assignment]
enhance_image = image_utils_module.enhance_image  # type: ignore[attr-defined]
image_to_base64 = image_utils_module.image_to_base64  # type: ignore[attr-defined]

plot_utils = import_module('utils.plot_utils')  # type: ignore[assignment]
plot_regression_scatter = plot_utils.plot_regression_scatter  # type: ignore[attr-defined]
plot_confusion_matrix = plot_utils.plot_confusion_matrix  # type: ignore[attr-defined]

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# ============ Health & Status ============
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'models': list_models(),
        'message': 'ML Integration Backend is running'
    })


# ============ Training Endpoints ============
@app.route('/train/linear', methods=['POST'])
def train_linear():
    """Train linear regression on single feature."""
    try:
        df = None
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename:
                df = load_csv_from_buffer(file)
        
        result = train_linear_regression(df)
        
        # Generate plot
        import numpy as np
        graph = plot_regression_scatter(
            np.array(result['y_test']),
            np.array(result['y_pred']),
            title='Linear Regression'
        )
        
        save_model('linear', result)
        
        return jsonify({
            'status': 'success',
            'model': 'linear_regression',
            'mse': result['mse'],
            'r2': result['r2'],
            'coefficients': result['coefficients'],
            'intercept': result['intercept'],
            'n_features': result['n_features'],
            'graph': graph
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/train/multiple', methods=['POST'])
def train_multiple():
    """Train multiple linear regression."""
    try:
        df = None
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename:
                df = load_csv_from_buffer(file)
        
        result = train_multiple_linear_regression(df)
        
        import numpy as np
        graph = plot_regression_scatter(
            np.array(result['y_test']),
            np.array(result['y_pred']),
            title='Multiple Linear Regression'
        )
        
        save_model('multiple', result)
        
        return jsonify({
            'status': 'success',
            'model': 'multiple_linear_regression',
            'mse': result['mse'],
            'r2': result['r2'],
            'coefficients': result['coefficients'],
            'intercept': result['intercept'],
            'n_features': result['n_features'],
            'graph': graph
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/train/logistic/binary', methods=['POST'])
def train_logistic_bin():
    """Train binary logistic regression."""
    try:
        df = None
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename:
                df = load_csv_from_buffer(file)
        
        result = train_logistic_binary(df)
        
        graph = plot_confusion_matrix(
            result['confusion_matrix'],
            title='Binary Logistic - Confusion Matrix'
        )
        
        save_model('logistic_binary', result)
        
        return jsonify({
            'status': 'success',
            'model': 'logistic_regression_binary',
            'accuracy': result['accuracy'],
            'confusion_matrix': result['confusion_matrix'],
            'n_features': result['n_features'],
            'graph': graph
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/train/logistic/multiclass', methods=['POST'])
def train_logistic_multi():
    """Train multiclass logistic regression."""
    try:
        df = None
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename:
                df = load_csv_from_buffer(file)
        
        result = train_logistic_multiclass(df)
        
        graph = plot_confusion_matrix(
            result['confusion_matrix'],
            title='Multiclass Logistic - Confusion Matrix'
        )
        
        save_model('logistic_multi', result)
        
        return jsonify({
            'status': 'success',
            'model': 'logistic_regression_multiclass',
            'accuracy': result['accuracy'],
            'confusion_matrix': result['confusion_matrix'],
            'n_features': result['n_features'],
            'graph': graph
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/train/dummy', methods=['POST'])
def train_dum():
    """Train dummy regression model."""
    try:
        df = None
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename:
                df = load_csv_from_buffer(file)
        
        result = train_dummy_regression(df)
        
        import numpy as np
        graph = plot_regression_scatter(
            np.array(result['y_test']),
            np.array(result['y_pred']),
            title='Dummy Regression Model'
        )
        
        save_model('dummy', result)
        
        return jsonify({
            'status': 'success',
            'model': 'dummy_regression',
            'mse': result['mse'],
            'r2': result['r2'],
            'coefficients': result['coefficients'],
            'n_features': result['n_features'],
            'graph': graph
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


# ============ Prediction Endpoint ============
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """Make a prediction using a trained model."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'status': 'error', 'message': 'No JSON data provided'}), 400
        
        model_name = data.get('model_name')
        input_list = data.get('input')
        
        if not model_name or not input_list:
            return jsonify({'status': 'error', 'message': 'model_name and input are required'}), 400
        
        model_data = get_model(model_name)
        prediction = predict(model_data, input_list)
        
        return jsonify({
            'status': 'success',
            'model': model_name,
            'prediction': prediction,
            'input': input_list
        })
    except ValueError as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ============ Image Enhancement Endpoint ============
@app.route('/enhance/image', methods=['POST'])
def enhance_endpoint():
    """Enhance an uploaded image."""
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file provided'}), 400
        
        file = request.files['file']
        
        if not file or not file.filename:
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400
        
        # Read image bytes
        image_bytes = file.read()
        
        # Enhance
        enhanced_img, uiqm, uciqe = enhance_image(image_bytes)
        
        # Convert to base64
        enhanced_b64 = image_to_base64(enhanced_img)
        
        return jsonify({
            'status': 'success',
            'enhanced_image': enhanced_b64,
            'uiqm': round(uiqm, 3),
            'uciqe': round(uciqe, 3),
            'note': 'Image is returned as base64 PNG. UIQM and UCIQE are quality metrics.'
        })
    except ValueError as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ============ Error Handlers ============
@app.errorhandler(404)
def not_found(error):
    return jsonify({'status': 'error', 'message': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
