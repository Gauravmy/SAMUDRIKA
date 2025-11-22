"""
Test script for Flask ML backend.
Run this after starting the Flask server: python test_api.py

Tests all endpoints and prints results to console.
"""
import requests
import sys

BASE_URL = 'http://127.0.0.1:5000'

# ANSI colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'


def print_result(test_name, success, message=''):
    """Print test result with color."""
    status = f'{GREEN}✓ PASS{RESET}' if success else f'{RED}✗ FAIL{RESET}'
    print(f'{status} {test_name}')
    if message:
        print(f'  {message}')


def test_health():
    """Test GET /health endpoint."""
    try:
        resp = requests.get(f'{BASE_URL}/health')
        if resp.status_code == 200:
            data = resp.json()
            print_result('GET /health', True, f"Models: {data.get('models', [])}")
            return True
        else:
            print_result('GET /health', False, f"Status: {resp.status_code}")
            return False
    except Exception as e:
        print_result('GET /health', False, str(e))
        return False


def test_train_linear():
    """Test POST /train/linear endpoint."""
    try:
        resp = requests.post(f'{BASE_URL}/train/linear')
        if resp.status_code == 200:
            data = resp.json()
            success = data.get('status') == 'success'
            msg = f"MSE: {data.get('mse')}, R2: {data.get('r2')}"
            print_result('POST /train/linear', success, msg)
            return success
        else:
            print_result('POST /train/linear', False, f"Status: {resp.status_code}")
            return False
    except Exception as e:
        print_result('POST /train/linear', False, str(e))
        return False


def test_train_multiple():
    """Test POST /train/multiple endpoint."""
    try:
        resp = requests.post(f'{BASE_URL}/train/multiple')
        if resp.status_code == 200:
            data = resp.json()
            success = data.get('status') == 'success'
            msg = f"MSE: {data.get('mse')}, R2: {data.get('r2')}, Features: {data.get('n_features')}"
            print_result('POST /train/multiple', success, msg)
            return success
        else:
            print_result('POST /train/multiple', False, f"Status: {resp.status_code}")
            return False
    except Exception as e:
        print_result('POST /train/multiple', False, str(e))
        return False


def test_train_logistic_binary():
    """Test POST /train/logistic/binary endpoint."""
    try:
        resp = requests.post(f'{BASE_URL}/train/logistic/binary')
        if resp.status_code == 200:
            data = resp.json()
            success = data.get('status') == 'success'
            msg = f"Accuracy: {data.get('accuracy')}"
            print_result('POST /train/logistic/binary', success, msg)
            return success
        else:
            print_result('POST /train/logistic/binary', False, f"Status: {resp.status_code}")
            return False
    except Exception as e:
        print_result('POST /train/logistic/binary', False, str(e))
        return False


def test_train_logistic_multiclass():
    """Test POST /train/logistic/multiclass endpoint."""
    try:
        resp = requests.post(f'{BASE_URL}/train/logistic/multiclass')
        if resp.status_code == 200:
            data = resp.json()
            success = data.get('status') == 'success'
            msg = f"Accuracy: {data.get('accuracy')}"
            print_result('POST /train/logistic/multiclass', success, msg)
            return success
        else:
            print_result('POST /train/logistic/multiclass', False, f"Status: {resp.status_code}")
            return False
    except Exception as e:
        print_result('POST /train/logistic/multiclass', False, str(e))
        return False


def test_train_dummy():
    """Test POST /train/dummy endpoint."""
    try:
        resp = requests.post(f'{BASE_URL}/train/dummy')
        if resp.status_code == 200:
            data = resp.json()
            success = data.get('status') == 'success'
            msg = f"MSE: {data.get('mse')}, R2: {data.get('r2')}"
            print_result('POST /train/dummy', success, msg)
            return success
        else:
            print_result('POST /train/dummy', False, f"Status: {resp.status_code}")
            return False
    except Exception as e:
        print_result('POST /train/dummy', False, str(e))
        return False


def test_predict():
    """Test POST /predict endpoint."""
    try:
        payload = {
            'model_name': 'linear',
            'input': [1.5]
        }
        resp = requests.post(
            f'{BASE_URL}/predict',
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        if resp.status_code == 200:
            data = resp.json()
            success = data.get('status') == 'success'
            msg = f"Prediction: {data.get('prediction')}"
            print_result('POST /predict', success, msg)
            return success
        else:
            print_result('POST /predict', False, f"Status: {resp.status_code}")
            return False
    except Exception as e:
        print_result('POST /predict', False, str(e))
        return False


def test_enhance_image():
    """Test POST /enhance/image endpoint."""
    try:
        # Create a simple test image (100x100 RGB)
        import numpy as np
        import cv2
        from io import BytesIO
        
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        success_encode, buffer = cv2.imencode('.jpg', test_img)
        
        if not success_encode:
            print_result('POST /enhance/image', False, 'Failed to create test image')
            return False
        
        files = {'file': ('test.jpg', BytesIO(buffer.tobytes()), 'image/jpeg')}
        resp = requests.post(f'{BASE_URL}/enhance/image', files=files)
        
        if resp.status_code == 200:
            data = resp.json()
            success = data.get('status') == 'success'
            msg = f"UIQM: {data.get('uiqm')}, UCIQE: {data.get('uciqe')}"
            print_result('POST /enhance/image', success, msg)
            return success
        else:
            print_result('POST /enhance/image', False, f"Status: {resp.status_code}")
            return False
    except Exception as e:
        print_result('POST /enhance/image', False, str(e))
        return False


def main():
    """Run all tests."""
    print(f'{YELLOW}=== Flask ML Backend Test Suite ==={RESET}\n')
    
    results = []
    
    results.append(('Health Check', test_health()))
    results.append(('Train Linear', test_train_linear()))
    results.append(('Train Multiple Linear', test_train_multiple()))
    results.append(('Train Logistic Binary', test_train_logistic_binary()))
    results.append(('Train Logistic Multiclass', test_train_logistic_multiclass()))
    results.append(('Train Dummy', test_train_dummy()))
    results.append(('Predict', test_predict()))
    results.append(('Enhance Image', test_enhance_image()))
    
    print(f'\n{YELLOW}=== Test Summary ==={RESET}')
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f'Passed: {passed}/{total}')
    
    if passed == total:
        print(f'{GREEN}All tests passed!{RESET}')
        return 0
    else:
        print(f'{RED}Some tests failed.{RESET}')
        return 1


if __name__ == '__main__':
    print('Make sure the Flask server is running: python -m flask --app app run')
    print('Waiting 2 seconds for server to be ready...\n')
    
    import time
    time.sleep(2)
    
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f'\n{YELLOW}Test interrupted by user.{RESET}')
        sys.exit(1)
