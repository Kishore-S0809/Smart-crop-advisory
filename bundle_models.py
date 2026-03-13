#!/usr/bin/env python3
"""
Model bundling script to select best performing models and archive the rest.
"""

import os
import json
import joblib
import shutil
from collections import defaultdict
from datetime import datetime

# Paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO_ROOT, 'models')
ARCHIVE_DIR = os.path.join(MODELS_DIR, 'archive')
REPORTS_DIR = os.path.join(REPO_ROOT, 'reports')
REPORT_JSON = os.path.join(REPORTS_DIR, 'report.json')
BUNDLE_PATH = os.path.join(MODELS_DIR, 'smart_crop_advisory.joblib')

def load_report():
    """Load the training report."""
    with open(REPORT_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)

def categorize_models(report):
    """Categorize models by purpose and find best performing ones."""
    models_by_purpose = defaultdict(list)
    
    for dataset in report['datasets']:
        if dataset['saved_model'] and dataset['metrics']:
            models_by_purpose[dataset['purpose']].append(dataset)
    
    return models_by_purpose

def select_best_model(models, purpose):
    """Select the best model for a given purpose based on metrics."""
    if not models:
        return None
    
    if purpose == 'crop_recommendation':
        # For classification: prefer highest accuracy, for regression: highest r2
        best = None
        best_score = -1
        for model in models:
            if model['model'] == 'classification':
                score = model['metrics'].get('accuracy', 0)
            elif model['model'] == 'regression':
                score = model['metrics'].get('r2', -1)
            else:
                continue
            
            if score > best_score:
                best_score = score
                best = model
        return best
    
    elif purpose == 'fertilizer_advisory':
        # For fertilizer: prefer highest accuracy/f1
        best = None
        best_score = -1
        for model in models:
            score = model['metrics'].get('accuracy', 0)
            if score > best_score:
                best_score = score
                best = model
        return best
    
    elif purpose == 'weather':
        # For weather: prefer lowest MAE
        best = None
        best_mae = float('inf')
        for model in models:
            mae = model['metrics'].get('mae', float('inf'))
            if mae < best_mae:
                best_mae = mae
                best = model
        return best
    
    return models[0]  # fallback

def bundle_models():
    """Main function to bundle best models and archive the rest."""
    
    # Load report
    report = load_report()
    
    # Categorize models
    models_by_purpose = categorize_models(report)
    
    print(f"Found models for purposes: {list(models_by_purpose.keys())}")
    
    # Select best models
    best_models = {}
    selected_files = set()
    
    for purpose, models in models_by_purpose.items():
        best = select_best_model(models, purpose)
        if best:
            best_models[purpose] = best
            selected_files.add(os.path.basename(best['saved_model']))
            print(f"Best {purpose} model: {os.path.basename(best['saved_model'])} "
                  f"(metrics: {best['metrics']})")
    
    # Load the best models into memory
    model_bundle = {}
    for purpose, model_info in best_models.items():
        model_path = model_info['saved_model']
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                model_bundle[purpose] = {
                    'model': model,
                    'metadata': {
                        'source_file': os.path.basename(model_path),
                        'metrics': model_info['metrics'],
                        'preprocessing': model_info['preprocessing'],
                        'model_type': model_info['model']
                    }
                }
                print(f"Loaded {purpose} model from {os.path.basename(model_path)}")
            except Exception as e:
                print(f"Error loading {model_path}: {e}")
                # Try to find an alternative model for this purpose
                if purpose == 'weather':
                    alternative_models = [m for m in models_by_purpose[purpose] if m != model_info]
                    for alt_model in alternative_models:
                        try:
                            alt_path = alt_model['saved_model']
                            if os.path.exists(alt_path):
                                model = joblib.load(alt_path)
                                model_bundle[purpose] = {
                                    'model': model,
                                    'metadata': {
                                        'source_file': os.path.basename(alt_path),
                                        'metrics': alt_model['metrics'],
                                        'preprocessing': alt_model['preprocessing'],
                                        'model_type': alt_model['model']
                                    }
                                }
                                print(f"Loaded alternative {purpose} model from {os.path.basename(alt_path)}")
                                # Update best_models to reflect the actually loaded model
                                best_models[purpose] = alt_model
                                selected_files.add(os.path.basename(alt_path))
                                break
                        except Exception as e2:
                            print(f"Alternative model {alt_path} also failed: {e2}")
                            continue
    
    # Save bundled models
    if model_bundle:
        joblib.dump(model_bundle, BUNDLE_PATH)
        print(f"Saved bundled models to {BUNDLE_PATH}")
    
    # Create archive directory
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    
    # Move non-selected models to archive
    archived_files = []
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith(('.joblib', '.pkl', '.h5')) and filename not in selected_files:
            if filename != 'smart_crop_advisory.joblib':
                src = os.path.join(MODELS_DIR, filename)
                dst = os.path.join(ARCHIVE_DIR, filename)
                shutil.move(src, dst)
                archived_files.append(f"models/archive/{filename}")
                print(f"Archived {filename}")
    
    # Update report
    report['deployment_info'] = {
        'deployment_model': 'models/smart_crop_advisory.joblib',
        'selected_models': {purpose: info['metadata'] for purpose, info in model_bundle.items()},
        'archived_models': archived_files,
        'bundle_created': datetime.now().isoformat()
    }
    
    # Save updated report
    with open(REPORT_JSON, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"Updated report with deployment info")
    print(f"Bundled {len(model_bundle)} models")
    print(f"Archived {len(archived_files)} models")
    
    return model_bundle, archived_files

if __name__ == '__main__':
    bundle_models()