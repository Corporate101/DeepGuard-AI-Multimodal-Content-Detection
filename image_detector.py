import cv2
import numpy as np
import time
from PIL import Image
import os

class ImageDetector:
    def __init__(self):
        self.initialized = True
    
    def analyze_image(self, image_path):
        """Analyze image for AI generation artifacts"""
        start_time = time.time()
        
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'is_ai_generated': False,
                    'confidence': 0.5,
                    'error': 'Could not load image',
                    'processing_time': time.time() - start_time
                }
            
            # Extract various image features
            features = self._extract_image_features(image)
            
            # Simple heuristic-based detection (would be replaced with actual ML model)
            ai_probability = self._calculate_ai_probability(features)
            confidence = abs(ai_probability - 0.5) * 2
            is_ai = ai_probability > 0.7
            
            return {
                'is_ai_generated': is_ai,
                'confidence': confidence,
                'ai_probability': ai_probability,
                'features': features,
                'processing_time': time.time() - start_time,
                'image_dimensions': f"{image.shape[1]}x{image.shape[0]}"
            }
            
        except Exception as e:
            return {
                'is_ai_generated': False,
                'confidence': 0.5,
                'error': f'Analysis error: {str(e)}',
                'processing_time': time.time() - start_time
            }
    
    def _extract_image_features(self, image):
        """Extract features from image for analysis"""
        # Convert to different color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        features = {}
        
        # Basic statistics
        features['brightness_mean'] = np.mean(gray)
        features['brightness_std'] = np.std(gray)
        features['contrast'] = gray.std()
        
        # Color features
        features['color_std_b'] = np.std(image[:, :, 0])  # Blue channel
        features['color_std_g'] = np.std(image[:, :, 1])  # Green channel
        features['color_std_r'] = np.std(image[:, :, 2])  # Red channel
        
        # Edge detection features
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        # Texture features using GLCM-like properties
        features['smoothness'] = 1 / (1 + features['contrast'])
        
        # Frequency domain analysis (simplified)
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
        features['high_freq_energy'] = np.mean(magnitude_spectrum[magnitude_spectrum > np.percentile(magnitude_spectrum, 90)])
        
        # Noise analysis
        features['noise_level'] = self._estimate_noise(gray)
        
        return features
    
    def _estimate_noise(self, gray_image):
        """Estimate noise level in image"""
        # Simple noise estimation using Laplacian variance
        return cv2.Laplacian(gray_image, cv2.CV_64F).var()
    
    def _calculate_ai_probability(self, features):
        """Calculate probability of image being AI-generated"""
        # This is a simplified heuristic - would be replaced with actual ML model
        
        score = 0.5  # Base probability
        
        # Adjust based on features (heuristic rules)
        if features.get('edge_density', 0) < 0.01:
            score += 0.2  # Very smooth images often AI-generated
        
        if features.get('noise_level', 0) < 100:
            score += 0.1  # Very clean images
        
        if features.get('high_freq_energy', 0) > 100:
            score -= 0.1  # High frequency content often indicates real photos
        
        # Normalize to [0, 1]
        return max(0, min(1, score))
