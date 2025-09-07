# BentoML Scikit-learn Integration

Comprehensive API reference for integrating Scikit-learn models with BentoML.

## Overview

BentoML provides seamless integration with Scikit-learn, supporting the full ecosystem of sklearn estimators, pipelines, and preprocessing tools.

## Core Functions

### save_model()

Save a Scikit-learn model to BentoML's model store.

```python
bentoml.sklearn.save_model(
    name: str,
    model: sklearn.base.BaseEstimator,
    signatures: dict = None,
    labels: dict = None,
    custom_objects: dict = None,
    metadata: dict = None
) -> bentoml.Tag
```

**Parameters:**
- `name` (str): Model name for identification
- `model` (sklearn.base.BaseEstimator): Scikit-learn model or pipeline
- `signatures` (dict, optional): Model signature configuration
- `labels` (dict, optional): Labels for model organization
- `custom_objects` (dict, optional): Additional objects to save
- `metadata` (dict, optional): Custom metadata

**Returns:** `bentoml.Tag` with model name and version

#### Basic Usage

```python
import bentoml
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset and train model
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
tag = bentoml.sklearn.save_model("iris_classifier", model)
print(f"Model saved: {tag}")
```

#### Advanced Configuration

```python
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

pipeline.fit(X_train, y_train)

# Calculate metrics
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save with comprehensive configuration
tag = bentoml.sklearn.save_model(
    name="iris_pipeline",
    model=pipeline,
    signatures={
        "predict": {
            "batchable": True,
            "batch_dim": 0
        },
        "predict_proba": {
            "batchable": True,
            "batch_dim": 0
        }
    },
    labels={
        "stage": "production",
        "algorithm": "random_forest",
        "task": "classification"
    },
    metadata={
        "accuracy": accuracy,
        "n_features": X_train.shape[1],
        "n_classes": len(iris.target_names),
        "sklearn_version": sklearn.__version__
    },
    custom_objects={
        "feature_names": iris.feature_names,
        "class_names": iris.target_names.tolist()
    }
)
```

### load_model()

Load a saved Scikit-learn model from BentoML's model store.

```python
bentoml.sklearn.load_model(
    tag: str
) -> sklearn.base.BaseEstimator
```

**Parameters:**
- `tag` (str): Model tag (name:version or name for latest)

**Returns:** Scikit-learn estimator or pipeline

#### Basic Loading

```python
# Load latest version
model = bentoml.sklearn.load_model("iris_classifier:latest")

# Load specific version
model = bentoml.sklearn.load_model("iris_classifier:v1.2.0")

# Use loaded model for prediction
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### get()

Get a BentoML model reference without loading into memory.

```python
model_ref = bentoml.sklearn.get("model_name:latest")
# Use model_ref.path for model files
# Use model_ref.info for metadata
# Use model_ref.custom_objects for additional objects
```

## Service Integration

### Basic Classification Service

```python
import bentoml
import numpy as np
from typing import List

@bentoml.service
class SklearnClassifier:
    def __init__(self):
        # Load model during service initialization
        self.model = bentoml.sklearn.load_model("iris_classifier:latest")
        
        # Load custom objects
        model_ref = bentoml.sklearn.get("iris_classifier:latest")
        self.feature_names = model_ref.custom_objects.get("feature_names", [])
        self.class_names = model_ref.custom_objects.get("class_names", [])
    
    @bentoml.api
    def predict(self, features: List[float]) -> dict:
        # Convert input to numpy array
        X = np.array([features])
        
        # Get prediction and probability
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        return {
            "prediction": int(prediction),
            "class_name": self.class_names[prediction] if self.class_names else None,
            "probabilities": {
                self.class_names[i] if self.class_names else f"class_{i}": float(prob)
                for i, prob in enumerate(probabilities)
            },
            "confidence": float(max(probabilities))
        }
    
    @bentoml.api
    def predict_batch(self, feature_batch: List[List[float]]) -> List[dict]:
        # Convert to numpy array
        X = np.array(feature_batch)
        
        # Batch prediction
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            results.append({
                "prediction": int(pred),
                "class_name": self.class_names[pred] if self.class_names else None,
                "confidence": float(max(probs))
            })
        
        return results
```

### Regression Service

```python
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

@bentoml.service
class SklearnRegressor:
    def __init__(self):
        self.model = bentoml.sklearn.load_model("house_price_regressor:latest")
        
        # Load feature information
        model_ref = bentoml.sklearn.get("house_price_regressor:latest")
        self.feature_names = model_ref.custom_objects.get("feature_names", [])
    
    @bentoml.api
    def predict_price(self, features: dict) -> dict:
        # Convert dict to DataFrame for consistent feature ordering
        df = pd.DataFrame([features])
        
        # Ensure correct feature order
        if self.feature_names:
            df = df.reindex(columns=self.feature_names, fill_value=0)
        
        # Make prediction
        prediction = self.model.predict(df.values)[0]
        
        # Get feature importance if available
        feature_importance = {}
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = {
                name: float(importance)
                for name, importance in zip(self.feature_names, self.model.feature_importances_)
            }
        
        return {
            "predicted_price": float(prediction),
            "feature_importance": feature_importance,
            "model_type": type(self.model).__name__
        }
```

### Pipeline Service

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

@bentoml.service
class SklearnPipeline:
    def __init__(self):
        self.pipeline = bentoml.sklearn.load_model("preprocessing_pipeline:latest")
        
        # Load metadata about expected features
        model_ref = bentoml.sklearn.get("preprocessing_pipeline:latest")
        self.numeric_features = model_ref.custom_objects.get("numeric_features", [])
        self.categorical_features = model_ref.custom_objects.get("categorical_features", [])
        self.expected_features = self.numeric_features + self.categorical_features
    
    @bentoml.api
    def predict_with_preprocessing(self, raw_data: dict) -> dict:
        # Validate input features
        missing_features = set(self.expected_features) - set(raw_data.keys())
        if missing_features:
            return {
                "error": f"Missing required features: {list(missing_features)}"
            }
        
        # Convert to DataFrame
        df = pd.DataFrame([raw_data])
        
        # Ensure correct column order
        df = df.reindex(columns=self.expected_features, fill_value=0)
        
        try:
            # Pipeline handles preprocessing and prediction
            prediction = self.pipeline.predict(df)[0]
            prediction_proba = self.pipeline.predict_proba(df)[0] if hasattr(self.pipeline, 'predict_proba') else None
            
            result = {
                "prediction": int(prediction) if isinstance(prediction, (np.integer, np.int64)) else float(prediction),
                "input_features": raw_data
            }
            
            if prediction_proba is not None:
                result["probabilities"] = prediction_proba.tolist()
                result["confidence"] = float(max(prediction_proba))
            
            return result
            
        except Exception as e:
            return {
                "error": f"Prediction failed: {str(e)}",
                "input_features": raw_data
            }
```

## Advanced Use Cases

### Multi-Model Ensemble

```python
@bentoml.service
class SklearnEnsemble:
    def __init__(self):
        # Load multiple models
        self.rf_model = bentoml.sklearn.load_model("random_forest:latest")
        self.gb_model = bentoml.sklearn.load_model("gradient_boosting:latest")
        self.svm_model = bentoml.sklearn.load_model("svm:latest")
        
        # Model weights for ensemble
        self.weights = [0.4, 0.4, 0.2]
    
    @bentoml.api
    def ensemble_predict(self, features: List[float]) -> dict:
        X = np.array([features])
        
        # Get predictions from all models
        rf_proba = self.rf_model.predict_proba(X)[0]
        gb_proba = self.gb_model.predict_proba(X)[0]
        svm_proba = self.svm_model.predict_proba(X)[0]
        
        # Weighted ensemble
        ensemble_proba = (
            self.weights[0] * rf_proba +
            self.weights[1] * gb_proba +
            self.weights[2] * svm_proba
        )
        
        ensemble_prediction = np.argmax(ensemble_proba)
        
        return {
            "ensemble_prediction": int(ensemble_prediction),
            "ensemble_confidence": float(max(ensemble_proba)),
            "individual_predictions": {
                "random_forest": {"prediction": int(np.argmax(rf_proba)), "confidence": float(max(rf_proba))},
                "gradient_boosting": {"prediction": int(np.argmax(gb_proba)), "confidence": float(max(gb_proba))},
                "svm": {"prediction": int(np.argmax(svm_proba)), "confidence": float(max(svm_proba))}
            }
        }
```

### Model Comparison Service

```python
@bentoml.service
class ModelComparison:
    def __init__(self):
        # Load different models for comparison
        self.models = {
            "linear": bentoml.sklearn.load_model("linear_model:latest"),
            "random_forest": bentoml.sklearn.load_model("rf_model:latest"),
            "gradient_boosting": bentoml.sklearn.load_model("gb_model:latest")
        }
    
    @bentoml.api
    def compare_models(self, features: List[float]) -> dict:
        X = np.array([features])
        results = {}
        
        for model_name, model in self.models.items():
            try:
                prediction = model.predict(X)[0]
                
                # Get probability if available
                probability = None
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[0]
                    probability = {
                        "probabilities": proba.tolist(),
                        "confidence": float(max(proba))
                    }
                
                results[model_name] = {
                    "prediction": int(prediction) if isinstance(prediction, (np.integer, np.int64)) else float(prediction),
                    "probability": probability,
                    "model_type": type(model).__name__
                }
                
            except Exception as e:
                results[model_name] = {
                    "error": str(e)
                }
        
        return {
            "comparisons": results,
            "input_features": features
        }
```

### Feature Importance Analysis

```python
@bentoml.service
class FeatureAnalyzer:
    def __init__(self):
        self.model = bentoml.sklearn.load_model("interpretable_model:latest")
        
        # Load feature metadata
        model_ref = bentoml.sklearn.get("interpretable_model:latest")
        self.feature_names = model_ref.custom_objects.get("feature_names", [])
    
    @bentoml.api
    def predict_with_explanation(self, features: List[float]) -> dict:
        X = np.array([features])
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        # Get feature importance
        feature_importance = {}
        if hasattr(self.model, 'feature_importances_'):
            importance_scores = self.model.feature_importances_
            feature_importance = {
                (self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"): float(score)
                for i, score in enumerate(importance_scores)
            }
        
        # Calculate prediction confidence
        confidence = None
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[0]
            confidence = float(max(proba))
        
        return {
            "prediction": int(prediction) if isinstance(prediction, (np.integer, np.int64)) else float(prediction),
            "confidence": confidence,
            "feature_importance": dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)),
            "top_features": list(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]),
            "input_features": dict(zip(self.feature_names[:len(features)], features))
        }
```

## Common Patterns

### Cross-Validation Results

```python
from sklearn.model_selection import cross_val_score

# Save model with cross-validation results
cv_scores = cross_val_score(model, X_train, y_train, cv=5)

tag = bentoml.sklearn.save_model(
    "validated_model",
    model,
    metadata={
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "cv_scores": cv_scores.tolist(),
        "validation_method": "5-fold_cv"
    }
)
```

### Grid Search Results

```python
from sklearn.model_selection import GridSearchCV

# Save best model from grid search
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

tag = bentoml.sklearn.save_model(
    "optimized_model",
    grid_search.best_estimator_,
    metadata={
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "cv_results": grid_search.cv_results_
    }
)
```

### Custom Transformers

```python
from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, param=1.0):
        self.param = param
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X * self.param

# Include custom transformer in pipeline
pipeline = Pipeline([
    ('custom', CustomTransformer(param=2.0)),
    ('classifier', RandomForestClassifier())
])

# Save pipeline with custom transformer
tag = bentoml.sklearn.save_model("custom_pipeline", pipeline)
```

## Best Practices

### 1. Model Validation

```python
@bentoml.service
class ValidatedService:
    def __init__(self):
        self.model = bentoml.sklearn.load_model("validated_model:latest")
        
        # Load model metadata for validation
        model_ref = bentoml.sklearn.get("validated_model:latest")
        self.expected_features = model_ref.metadata.get("n_features", 0)
    
    @bentoml.api
    def predict(self, features: List[float]) -> dict:
        # Validate input dimensions
        if len(features) != self.expected_features:
            return {
                "error": f"Expected {self.expected_features} features, got {len(features)}"
            }
        
        # Validate feature values
        if any(not isinstance(f, (int, float)) for f in features):
            return {
                "error": "All features must be numeric"
            }
        
        X = np.array([features])
        prediction = self.model.predict(X)[0]
        
        return {
            "prediction": int(prediction) if isinstance(prediction, (np.integer, np.int64)) else float(prediction)
        }
```

### 2. Error Handling

```python
@bentoml.service
class RobustSklearnService:
    def __init__(self):
        try:
            self.model = bentoml.sklearn.load_model("robust_model:latest")
            self.model_loaded = True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_loaded = False
    
    @bentoml.api
    def predict(self, features: List[float]) -> dict:
        if not self.model_loaded:
            return {"error": "Model not available"}
        
        try:
            X = np.array([features])
            prediction = self.model.predict(X)[0]
            
            return {
                "success": True,
                "prediction": int(prediction) if isinstance(prediction, (np.integer, np.int64)) else float(prediction)
            }
            
        except ValueError as e:
            return {"error": f"Invalid input: {str(e)}"}
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": "Prediction failed"}
```

### 3. Performance Monitoring

```python
import time

@bentoml.service
class MonitoredService:
    def __init__(self):
        self.model = bentoml.sklearn.load_model("monitored_model:latest")
        self.prediction_count = 0
        self.total_time = 0
    
    @bentoml.api
    def predict(self, features: List[float]) -> dict:
        start_time = time.time()
        
        try:
            X = np.array([features])
            prediction = self.model.predict(X)[0]
            
            # Update metrics
            prediction_time = time.time() - start_time
            self.prediction_count += 1
            self.total_time += prediction_time
            
            return {
                "prediction": int(prediction) if isinstance(prediction, (np.integer, np.int64)) else float(prediction),
                "prediction_time": prediction_time,
                "total_predictions": self.prediction_count,
                "average_time": self.total_time / self.prediction_count
            }
            
        except Exception as e:
            return {"error": str(e)}
```

For more Scikit-learn examples, visit the [BentoML examples repository](https://github.com/bentoml/BentoML/tree/main/examples/).