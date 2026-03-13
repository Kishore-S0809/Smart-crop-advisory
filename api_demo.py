#!/usr/bin/env python3
"""
Smart Crop Advisory API - Example usage of bundled models.
"""

import os
import joblib
import pandas as pd
from typing import Dict, Any, Optional

# Paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_PATH = os.path.join(REPO_ROOT, 'models', 'smart_crop_advisory.joblib')

class SmartCropAdvisory:
    """Smart Crop Advisory system using bundled models."""
    
    def __init__(self, bundle_path: str = BUNDLE_PATH):
        """Initialize the advisory system by loading the bundled models."""
        self.models = {}
        self.metadata = {}
        
        if os.path.exists(bundle_path):
            bundle = joblib.load(bundle_path)
            for purpose, model_info in bundle.items():
                self.models[purpose] = model_info['model']
                self.metadata[purpose] = model_info['metadata']
                print(f"Loaded {purpose} model: {model_info['metadata']['source_file']}")
        else:
            print(f"Warning: Bundle file not found at {bundle_path}")
    
    def get_crop_recommendation(self, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get crop recommendation based on environmental features."""
        if 'crop_recommendation' not in self.models:
            return {"error": "Crop recommendation model not available"}
        
        try:
            # Convert features to DataFrame for preprocessing
            df = pd.DataFrame([features])
            model = self.models['crop_recommendation']
            prediction = model.predict(df)
            probabilities = model.predict_proba(df) if hasattr(model, 'predict_proba') else None
            
            result = {
                "prediction": prediction[0] if len(prediction) > 0 else None,
                "confidence": float(max(probabilities[0])) if probabilities is not None else None,
                "model_info": self.metadata['crop_recommendation']
            }
            return result
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def get_fertilizer_recommendation(self, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get fertilizer recommendation based on soil and crop features."""
        if 'fertilizer_advisory' not in self.models:
            return {"error": "Fertilizer advisory model not available"}
        
        try:
            # Convert features to DataFrame for preprocessing
            df = pd.DataFrame([features])
            model = self.models['fertilizer_advisory']
            prediction = model.predict(df)
            probabilities = model.predict_proba(df) if hasattr(model, 'predict_proba') else None
            
            result = {
                "prediction": prediction[0] if len(prediction) > 0 else None,
                "confidence": float(max(probabilities[0])) if probabilities is not None else None,
                "model_info": self.metadata['fertilizer_advisory']
            }
            return result
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def get_weather_forecast(self, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get weather forecast (if weather model is available)."""
        if 'weather' not in self.models:
            return {"error": "Weather model not available"}
        
        try:
            # Weather models typically require time series data
            model = self.models['weather']
            # This would need specific implementation based on the model type
            return {
                "message": "Weather model loaded but requires time series implementation",
                "model_info": self.metadata['weather']
            }
        except Exception as e:
            return {"error": f"Weather prediction failed: {str(e)}"}
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get information about available models."""
        return {
            "available_models": list(self.models.keys()),
            "metadata": self.metadata
        }

def demo_usage():
    """Demonstrate how to use the Smart Crop Advisory system."""
    print("=== Smart Crop Advisory Demo ===")
    
    # Initialize the advisory system
    advisory = SmartCropAdvisory()
    
    # Show available models
    print("\nAvailable models:")
    info = advisory.get_available_models()
    for model_name in info['available_models']:
        metadata = info['metadata'][model_name]
        print(f"  - {model_name}: {metadata['source_file']} (accuracy: {metadata['metrics'].get('accuracy', 'N/A')})")
    
    # Example crop recommendation (using apples dataset structure)
    print("\n=== Crop Recommendation Example ===")
    crop_features = {
        'Domain Code': 'QCL',
        'Domain': 'Crops and livestock products',
        'Area Code (FAO)': 100,
        'Area': 'India',
        'Element Code': 5312,
        'Element': 'Area harvested',
        'Item Code (FAO)': 515,
        'Item': 'Apples',
        'Year Code': 2020,
        'Year': 2020,
        'Unit': 'ha',
        'Value': 45000,
        'Flag': 'F',
        'Flag Description': 'FAO estimate'
    }
    
    crop_result = advisory.get_crop_recommendation(crop_features)
    print(f"Crop recommendation: {crop_result}")
    
    # Example fertilizer recommendation (using dataset.csv structure)
    print("\n=== Fertilizer Recommendation Example ===")
    fertilizer_features = {
        'N': 143,
        'P': 69,
        'K': 217,
        'ph': 5.9,
        'EC': 0.58,
        'S': 0.23,
        'Cu': 10.2,
        'Fe': 116.35,
        'Mn': 59.96,
        'Zn': 54.85,
        'B': 21.29
    }
    
    fertilizer_result = advisory.get_fertilizer_recommendation(fertilizer_features)
    print(f"Fertilizer recommendation: {fertilizer_result}")

if __name__ == '__main__':
    demo_usage()