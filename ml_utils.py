"""
ML utilities: training, prediction, metrics, and dummy data generation.
"""
import io
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_regression


def generate_dummy_regression(n_samples=200, n_features=1):
    """Generate dummy regression dataset."""
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=10.0, random_state=42)
    cols = [f"x{i+1}" for i in range(n_features)] + ["y"]
    data = np.hstack([X, y.reshape(-1, 1)])
    return pd.DataFrame(data, columns=cols)


def generate_dummy_classification(n_samples=200, n_features=2, n_classes=2):
    """Generate dummy classification dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_classes=n_classes,
        random_state=42
    )
    cols = [f"x{i+1}" for i in range(n_features)] + ["y"]
    data = np.hstack([X, y.reshape(-1, 1)])
    return pd.DataFrame(data, columns=cols)


def load_csv_from_buffer(file_stream) -> pd.DataFrame:
    """Load CSV from file stream (werkzeug FileStorage)."""
    try:
        stream = io.BytesIO(file_stream.read())
        df = pd.read_csv(stream)
        return df
    except Exception as e:
        raise ValueError(f"Failed to parse CSV: {str(e)}")


def split_xy_from_df(df: pd.DataFrame):
    """Split DataFrame into X (features) and y (target). Target is the last column."""
    if df.shape[1] < 2:
        raise ValueError("Dataset must have at least one feature and one target column")
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values
    return X, y


def train_linear_regression(df=None):
    """Train a single-feature linear regression."""
    if df is None:
        df = generate_dummy_regression(n_samples=200, n_features=1)
    
    X, y = split_xy_from_df(df)
    # Use only first feature for linear regression
    if X.ndim == 2 and X.shape[1] > 1:
        X = X[:, [0]]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = float(mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    
    return {
        'model': model,
        'mse': mse,
        'r2': r2,
        'coefficients': model.coef_.flatten().tolist(),
        'intercept': float(model.intercept_),
        'y_test': y_test.tolist(),
        'y_pred': y_pred.tolist(),
        'n_features': X_train.shape[1]
    }


def train_multiple_linear_regression(df=None):
    """Train multiple-feature linear regression."""
    if df is None:
        df = generate_dummy_regression(n_samples=300, n_features=3)
    
    X, y = split_xy_from_df(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = float(mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    
    return {
        'model': model,
        'mse': mse,
        'r2': r2,
        'coefficients': model.coef_.flatten().tolist(),
        'intercept': float(model.intercept_),
        'y_test': y_test.tolist(),
        'y_pred': y_pred.tolist(),
        'n_features': X_train.shape[1]
    }


def train_logistic_binary(df=None):
    """Train binary logistic regression."""
    if df is None:
        df = generate_dummy_classification(n_samples=250, n_features=2, n_classes=2)
    
    X, y = split_xy_from_df(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    return {
        'model': model,
        'scaler': scaler,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'y_test': y_test.tolist(),
        'y_pred': y_pred.tolist(),
        'n_features': X_train.shape[1]
    }


def train_logistic_multiclass(df=None):
    """Train multiclass logistic regression."""
    if df is None:
        df = generate_dummy_classification(n_samples=300, n_features=4, n_classes=3)
    
    X, y = split_xy_from_df(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    return {
        'model': model,
        'scaler': scaler,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'y_test': y_test.tolist(),
        'y_pred': y_pred.tolist(),
        'n_features': X_train.shape[1]
    }


def train_dummy_regression(df=None):
    """Train a dummy model: y = 2*x + random noise."""
    if df is None:
        df = generate_dummy_regression(n_samples=200, n_features=1)
    
    X, y = split_xy_from_df(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Simple rule: y = 2 * X[:, 0] + intercept
    if X.shape[1] > 0:
        coef = 2.0
        intercept = np.mean(y_train - coef * X_train[:, 0])
        y_pred = coef * X_test[:, 0] + intercept + np.random.normal(0, 1, len(y_test))
    else:
        y_pred = np.ones_like(y_test) * np.mean(y_train)
    
    mse = float(mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    
    def predict_fn(X_input):
        """Simple prediction function."""
        return 2.0 * X_input[:, 0] + np.mean(y_train)
    
    return {
        'model': predict_fn,
        'mse': mse,
        'r2': r2,
        'coefficients': [2.0],
        'y_test': y_test.tolist(),
        'y_pred': y_pred.tolist(),
        'n_features': X_train.shape[1]
    }


def predict(model_data, input_list):
    """Make a prediction using a trained model."""
    if model_data is None:
        raise ValueError("Model not found. Train it first.")
    
    model = model_data.get('model')
    scaler = model_data.get('scaler')
    n_features = model_data.get('n_features')
    
    if len(input_list) != n_features:
        raise ValueError(f"Expected {n_features} features, got {len(input_list)}")
    
    X = np.array(input_list).reshape(1, -1).astype(float)
    
    if scaler:
        X = scaler.transform(X)
    
    # Handle callable (dummy model) or sklearn model
    if callable(model):
        pred = model(X)
    else:
        pred = model.predict(X)
    
    # Convert to Python scalar
    if hasattr(pred, '__len__'):
        result = float(pred[0]) if len(pred) > 0 else float(pred)
    else:
        result = float(pred)
    
    return result
