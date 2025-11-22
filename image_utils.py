"""
Image enhancement utilities: CLAHE, color balance, denoising, sharpening, and quality metrics.
"""
import cv2
import numpy as np
from io import BytesIO
import base64


def enhance_image(image_bytes) -> tuple:
    """
    Full enhancement pipeline:
    1. Decode image from bytes
    2. Convert to LAB color space
    3. Apply CLAHE to L channel
    4. Apply color balance
    5. Apply bilateral filter (denoise)
    6. Apply sharpening
    7. Convert back to RGB
    8. Compute quality metrics
    
    Returns: (enhanced_image_bgr, uiqm, uciqe)
    """
    try:
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image")
        
        # Ensure image is uint8
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
        
        # Step 1: Convert to LAB
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(img_lab)
        
        # Step 2: Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        L_clahe = clahe.apply(L)
        
        # Step 3: Color balance (stretch AB channels)
        A_mean = np.mean(A)
        B_mean = np.mean(B)
        A_balanced = cv2.convertScaleAbs(A - A_mean) + A_mean
        B_balanced = cv2.convertScaleAbs(B - B_mean) + B_mean
        
        # Merge back to LAB
        img_lab_enhanced = cv2.merge([L_clahe, A_balanced.astype(np.uint8), B_balanced.astype(np.uint8)])
        
        # Step 4: Convert back to BGR
        img_bgr = cv2.cvtColor(img_lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Step 5: Bilateral filter (denoise while preserving edges)
        img_denoised = cv2.bilateralFilter(img_bgr, d=7, sigmaColor=20, sigmaSpace=20)
        
        # Step 6: Sharpening kernel
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]]) / 1.0
        img_sharpened = cv2.filter2D(img_denoised, -1, kernel)
        
        # Clip to valid range
        img_enhanced = np.clip(img_sharpened, 0, 255).astype(np.uint8)
        
        # Compute metrics on the enhanced image
        uiqm = compute_uiqm(img_enhanced)
        uciqe = compute_uciqe(img_enhanced)
        
        return img_enhanced, uiqm, uciqe
    
    except Exception as e:
        raise ValueError(f"Image enhancement failed: {str(e)}")


def compute_uiqm(image) -> float:
    """
    Underwater Image Quality Measure (approximation).
    Combines contrast, saturation, and sharpness.
    """
    # Convert to float
    img = image.astype(np.float32) / 255.0
    
    # Compute local contrast
    img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    contrast = np.std(img_gray)
    
    # Compute saturation (in HSV)
    img_hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    saturation = np.mean(img_hsv[:, :, 1])
    
    # Compute sharpness (Laplacian variance)
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    sharpness = np.var(laplacian)
    
    # Weighted combination
    uiqm = 0.4 * contrast + 0.3 * (saturation / 255.0) + 0.3 * (sharpness / 1000.0)
    return float(np.clip(uiqm, 0, 5))


def compute_uciqe(image) -> float:
    """
    Underwater Color Image Quality Evaluation (approximation).
    Based on color information and contrast.
    """
    img = image.astype(np.float32) / 255.0
    
    # Extract color channels
    b, g, r = cv2.split(img)
    
    # RG component
    rg = r - g
    rg_mean = np.mean(rg)
    rg_std = np.std(rg)
    
    # YB component
    yb = (r + g) / 2.0 - b
    yb_mean = np.mean(yb)
    yb_std = np.std(yb)
    
    # Color difference
    color_diff = np.sqrt(rg_mean**2 + yb_mean**2)
    
    # Saturation
    saturation = np.sqrt(rg_std**2 + yb_std**2)
    
    # Contrast
    img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    contrast = np.std(img_gray)
    
    # Combined metric
    uciqe = 0.4 * color_diff + 0.4 * saturation + 0.2 * (contrast / 255.0)
    return float(np.clip(uciqe, 0, 1))


def image_to_base64(image_bgr) -> str:
    """Convert OpenCV BGR image to base64 PNG string."""
    success, buffer = cv2.imencode('.png', image_bgr)
    if not success:
        raise ValueError("Failed to encode image to PNG")
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    return img_b64


def base64_to_image(b64_str):
    """Convert base64 string back to image bytes."""
    img_bytes = base64.b64decode(b64_str)
    return img_bytes
