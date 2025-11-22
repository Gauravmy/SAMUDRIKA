"""
Plot utilities: generate matplotlib scatter plots as base64 PNG.
"""
import io
import base64
import matplotlib.pyplot as plt
import numpy as np


def plot_regression_scatter(y_test, y_pred, title="Regression: y_test vs y_pred") -> str:
    """Create scatter plot of y_test vs y_pred and return as base64 PNG."""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    
    ax.scatter(y_test, y_pred, alpha=0.6, s=50, color='#00B4D8', edgecolors='#0077B6', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
    
    ax.set_xlabel('y_test', fontsize=11, fontweight='bold')
    ax.set_ylabel('y_pred', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_b64


def plot_confusion_matrix(cm, title="Confusion Matrix") -> str:
    """Create heatmap of confusion matrix and return as base64 PNG."""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    
    cm_array = np.array(cm)
    im = ax.imshow(cm_array, interpolation='nearest', cmap='Blues')
    
    # Add text annotations
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            ax.text(j, i, str(cm_array[i, j]), ha='center', va='center', 
                   color='white' if cm_array[i, j] > cm_array.max() / 2 else 'black', 
                   fontweight='bold')
    
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(np.arange(cm_array.shape[1]))
    ax.set_yticks(np.arange(cm_array.shape[0]))
    
    fig.colorbar(im, ax=ax)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_b64
