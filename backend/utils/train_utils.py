"""
Training utilities: loading data, generating dummy data, training models and producing metrics + graphs.
"""
from typing import Tuple, Optional
import io
import base64

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt

# --- Data loading / generation ---

def load_csv_from_upload(file) -> pd.DataFrame:
    """Read uploaded CSV (UploadFile) into a pandas DataFrame."""
    contents = file.file.read()
    file.file.close()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        # try reading as latin-1 or with pandas autodetect
        df = pd.read_csv(io.StringIO(contents.decode('utf-8', errors='replace')))
    return df


def generate_dummy_regression(n_samples: int = 200, n_features: int = 1) -> pd.DataFrame:
    """Generate a simple linear dataset. Last column is the target."""
    rng = np.random.RandomState(42)
    X = rng.uniform(-10, 10, size=(n_samples, n_features))
    # coefs
    coefs = rng.uniform(1.0, 3.0, size=(n_features,))
    y = X.dot(coefs) + rng.normal(scale=2.0, size=(n_samples,))
    cols = [f"x{i+1}" for i in range(n_features)] + ["y"]
    data = np.hstack([X, y.reshape(-1, 1)])
    return pd.DataFrame(data, columns=cols)


def generate_dummy_classification(n_samples: int = 200, n_features: int = 2, n_classes: int = 2) -> pd.DataFrame:
    """Generate a simple classification dataset using gaussian blobs."""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_features, n_redundant=0, n_classes=n_classes, random_state=42)
    cols = [f"x{i+1}" for i in range(n_features)] + ["y"]
    data = np.hstack([X, y.reshape(-1, 1)])
    return pd.DataFrame(data, columns=cols)


# --- Training functions ---

def split_xy_from_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Assumes the target is the last column. Returns X, y numpy arrays."""
    if df.shape[1] < 2:
        raise ValueError("Dataset must have at least one feature column and one target column")
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(float)
    return X, y


def create_scatter_png_base64(y_true: np.ndarray, y_pred: np.ndarray, title: Optional[str] = None) -> str:
    """Creates a scatter plot y_true vs y_pred and returns a base64 PNG string."""
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.scatter(y_true, y_pred, alpha=0.7, s=30, color='#00B4D8')
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color='#0077B6', linewidth=1, linestyle='--')
    ax.set_xlabel('y_test')
    ax.set_ylabel('y_pred')
    if title:
        ax.set_title(title)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_b64


def train_linear_regression(df: pd.DataFrame) -> dict:
    """Train a single-feature linear regression. If multiple features present, uses only the first column as feature.
    Returns a dict with model, metrics, coeffs, preds, graph.
    """
    X, y = split_xy_from_df(df)
    # ensure single feature
    if X.ndim == 2 and X.shape[1] > 1:
        X = X[:, [0]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = float(mean_squared_error(y_test, y_pred))
    # 'accuracy' is not typical for regression - we provide R2 as a proxy in 'accuracy' field for UI convenience
    try:
        from sklearn.metrics import r2_score
        r2 = float(r2_score(y_test, y_pred))
    except Exception:
        r2 = None
    coeffs = model.coef_.flatten().tolist()
    sample_predictions = [float(x) for x in y_pred[:5]]
    graph = create_scatter_png_base64(y_test, y_pred, title='Linear Regression: y_test vs y_pred')
    return {
        'model_obj': model,
        'mse': mse,
        'accuracy': r2,
        'coefficients': coeffs,
        'sample_predictions': sample_predictions,
        'graph': graph,
        'n_features': X_train.shape[1]
    }


def train_multiple_linear_regression(df: pd.DataFrame) -> dict:
    """Train linear regression on multiple features."""
    X, y = split_xy_from_df(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = float(mean_squared_error(y_test, y_pred))
    try:
        from sklearn.metrics import r2_score
        r2 = float(r2_score(y_test, y_pred))
    except Exception:
        r2 = None
    coeffs = model.coef_.flatten().tolist()
    sample_predictions = [float(x) for x in y_pred[:5]]
    graph = create_scatter_png_base64(y_test, y_pred, title='Multiple Linear Regression: y_test vs y_pred')
    return {
        'model_obj': model,
        'mse': mse,
        'accuracy': r2,
        'coefficients': coeffs,
        'sample_predictions': sample_predictions,
        'graph': graph,
        'n_features': X_train.shape[1]
    }


def train_logistic_classification(df: pd.DataFrame, multiclass: bool = False) -> dict:
    """Train logistic regression; multiclass if requested."""
    X, y = split_xy_from_df(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None)
    if multiclass:
        solver = 'lbfgs'
        multi = 'multinomial'
    else:
        solver = 'liblinear'
        multi = 'ovr'
    model = LogisticRegression(solver=solver, multi_class=multi, max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    coeffs = None
    try:
        coeffs = model.coef_.tolist()
    except Exception:
        coeffs = None
    sample_predictions = [int(x) if hasattr(x, '__int__') else x for x in y_pred[:5]]

    # For classification, create a jittered scatter for visualization
    y_test_v = y_test
    y_pred_v = y_pred
    graph = create_scatter_png_base64(y_test_v, y_pred_v, title='Logistic: y_test vs y_pred')

    return {
        'model_obj': model,
        'mse': None,
        'accuracy': acc,
        'coefficients': coeffs,
        'sample_predictions': sample_predictions,
        'graph': graph,
        'n_features': X_train.shape[1]
    }


def train_dummy_model(df: Optional[pd.DataFrame] = None) -> dict:
    """Create a dummy model: for regression returns y = 2*x1 + 1, for classification returns threshold on sum of features.
    If df is provided we'll infer type from target values (continuous vs discrete).
    """
    if df is None:
        df = generate_dummy_regression(n_samples=200, n_features=1)
    X, y = split_xy_from_df(df)
    # If target has many unique values, treat as regression
    if len(np.unique(y)) > 10:
        # regression dummy
        def predict_fn(X_in):
            return 2.0 * X_in[:, 0] + 1.0
        # make predictions on a test split
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_pred = predict_fn(X_test)
        mse = float(mean_squared_error(y_test, y_pred))
        graph = create_scatter_png_base64(y_test, y_pred, title='Dummy Regression: y_test vs y_pred')
        return {
            'model_obj': {'predict': predict_fn},
            'mse': mse,
            'accuracy': None,
            'coefficients': [2.0],
            'sample_predictions': [float(x) for x in y_pred[:5]],
            'graph': graph,
            'n_features': X.shape[1]
        }
    else:
        # classification dummy
        def predict_fn(X_in):
            s = X_in.sum(axis=1)
            return (s > np.median(s)).astype(int)
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_pred = predict_fn(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        graph = create_scatter_png_base64(y_test, y_pred, title='Dummy Classification: y_test vs y_pred')
        return {
            'model_obj': {'predict': predict_fn},
            'mse': None,
            'accuracy': acc,
            'coefficients': None,
            'sample_predictions': [int(x) for x in y_pred[:5]],
            'graph': graph,
            'n_features': X.shape[1]
        }
