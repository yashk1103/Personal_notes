# Stacking vs Voting Ensemble: Deep Dive Analysis

## Introduction

**Stacking** and **Voting Ensemble** are the two most popular ensemble methods in modern machine learning. While both combine multiple models, they differ fundamentally in **how** they combine predictions and **what** they learn from the combination process.

---

# PART I: VOTING ENSEMBLE

## Conceptual Foundation

Voting ensemble is based on the **democratic principle** - multiple models "vote" on the final prediction. It's the simplest form of ensemble learning, yet surprisingly effective.

### Core Assumption
If we have M models where each has accuracy > 50%, the majority vote will have higher accuracy than any individual model.

### Mathematical Proof

For binary classification with M odd models, each with error rate ε < 0.5:

**Probability of ensemble error** = P(majority wrong)

```
P(ensemble error) = Σ(k=⌈M/2⌉ to M) C(M,k) * ε^k * (1-ε)^(M-k)
```

This decreases exponentially as M increases!

**Example**: With 3 models, each 70% accurate (ε = 0.3):
- Individual accuracy: 70%
- Ensemble accuracy: 1 - [C(3,2)×0.3²×0.7¹ + C(3,3)×0.3³×0.7⁰] = 1 - [0.189 + 0.027] = 78.4%

## Types of Voting Ensemble

### 1. Hard Voting (Classification)

Each model outputs a discrete class label. Final prediction = majority vote.

#### Implementation Details

```python
class HardVotingClassifier:
    def __init__(self, models):
        self.models = models
        self.n_models = len(models)
        self.classes_ = None
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        # Train each model independently
        for model in self.models:
            model.fit(X, y)
        return self
    
    def predict(self, X):
        # Collect predictions from all models
        predictions = np.zeros((X.shape[0], self.n_models))
        
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
        
        # Majority vote for each sample
        final_predictions = []
        for i in range(X.shape[0]):
            votes = predictions[i, :]
            # Count votes for each class
            vote_counts = np.bincount(votes.astype(int))
            # Return class with most votes
            final_predictions.append(np.argmax(vote_counts))
        
        return np.array(final_predictions)
    
    def predict_confidence(self, X):
        """Return confidence as proportion of models agreeing"""
        predictions = np.zeros((X.shape[0], self.n_models))
        
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
        
        confidences = []
        final_predictions = []
        
        for i in range(X.shape[0]):
            votes = predictions[i, :]
            vote_counts = np.bincount(votes.astype(int))
            winner = np.argmax(vote_counts)
            confidence = vote_counts[winner] / self.n_models
            
            final_predictions.append(winner)
            confidences.append(confidence)
        
        return np.array(final_predictions), np.array(confidences)
```

#### Advanced Hard Voting Techniques

**Weighted Hard Voting**:
```python
def weighted_hard_voting(predictions, weights):
    """Each model's vote has different weight"""
    weighted_votes = predictions * weights[:, np.newaxis]
    return np.argmax(np.sum(weighted_votes, axis=0))
```

**Threshold-based Voting**:
```python
def threshold_voting(predictions, thresholds):
    """Require minimum threshold of votes to make prediction"""
    vote_counts = np.bincount(predictions)
    max_votes = np.max(vote_counts)
    
    if max_votes >= thresholds:
        return np.argmax(vote_counts)
    else:
        return "uncertain"  # Abstain from prediction
```

### 2. Soft Voting (Classification)

Uses predicted probabilities instead of hard class labels. More nuanced than hard voting.

#### Mathematical Foundation

For K classes and M models:
```
P(class_k | x) = (1/M) * Σ(i=1 to M) P_i(class_k | x)
```

Final prediction = argmax(averaged probabilities)

#### Implementation

```python
class SoftVotingClassifier:
    def __init__(self, models):
        self.models = models
        self.n_models = len(models)
        self.classes_ = None
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for model in self.models:
            model.fit(X, y)
        return self
    
    def predict_proba(self, X):
        """Return averaged probabilities"""
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # Collect probabilities from all models
        all_probas = np.zeros((n_samples, n_classes, self.n_models))
        
        for i, model in enumerate(self.models):
            probas = model.predict_proba(X)
            all_probas[:, :, i] = probas
        
        # Average probabilities across models
        avg_probas = np.mean(all_probas, axis=2)
        return avg_probas
    
    def predict(self, X):
        avg_probas = self.predict_proba(X)
        return self.classes_[np.argmax(avg_probas, axis=1)]
    
    def predict_with_entropy(self, X):
        """Return predictions with uncertainty measure"""
        avg_probas = self.predict_proba(X)
        
        # Calculate entropy as uncertainty measure
        entropy = -np.sum(avg_probas * np.log(avg_probas + 1e-10), axis=1)
        predictions = self.classes_[np.argmax(avg_probas, axis=1)]
        
        return predictions, entropy
```

#### Advanced Soft Voting Techniques

**Temperature Scaling**:
```python
def temperature_scaled_voting(probas_list, temperatures):
    """Apply temperature scaling before averaging"""
    scaled_probas = []
    
    for i, probas in enumerate(probas_list):
        # Apply temperature scaling
        scaled_logits = np.log(probas + 1e-10) / temperatures[i]
        # Convert back to probabilities
        scaled_probas.append(softmax(scaled_logits))
    
    return np.mean(scaled_probas, axis=0)
```

**Confidence-weighted Voting**:
```python
def confidence_weighted_voting(probas_list, X, models):
    """Weight models by their confidence on each sample"""
    weighted_probas = []
    
    for i, (probas, model) in enumerate(zip(probas_list, models)):
        # Calculate confidence (max probability)
        confidence = np.max(probas, axis=1)
        
        # Weight probabilities by confidence
        weighted_probas.append(probas * confidence[:, np.newaxis])
    
    # Normalize weights
    total_weights = np.sum([np.max(p, axis=1) for p in probas_list], axis=0)
    
    final_probas = np.sum(weighted_probas, axis=0)
    final_probas /= total_weights[:, np.newaxis]
    
    return final_probas
```

### 3. Averaging (Regression)

For regression tasks, we average the predicted continuous values.

#### Simple Averaging

```python
class AveragingRegressor:
    def __init__(self, models):
        self.models = models
    
    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self
    
    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.models)))
        
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
        
        return np.mean(predictions, axis=1)
    
    def predict_with_std(self, X):
        """Return predictions with standard deviation"""
        predictions = np.zeros((X.shape[0], len(self.models)))
        
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
        
        mean_pred = np.mean(predictions, axis=1)
        std_pred = np.std(predictions, axis=1)
        
        return mean_pred, std_pred
```

#### Weighted Averaging

```python
class WeightedAveragingRegressor:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights
        self.optimized_weights = None
    
    def fit(self, X, y):
        # Train all models
        for model in self.models:
            model.fit(X, y)
        
        # If no weights provided, optimize them
        if self.weights is None:
            self.optimized_weights = self._optimize_weights(X, y)
        else:
            self.optimized_weights = np.array(self.weights)
            
        return self
    
    def _optimize_weights(self, X, y):
        """Find optimal weights using cross-validation"""
        from scipy.optimize import minimize
        
        def objective(weights):
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Calculate weighted prediction
            predictions = np.zeros(X.shape[0])
            for i, model in enumerate(self.models):
                pred = model.predict(X)
                predictions += weights[i] * pred
            
            # Return mean squared error
            return np.mean((predictions - y) ** 2)
        
        # Initialize with equal weights
        initial_weights = np.ones(len(self.models)) / len(self.models)
        
        # Constraints: weights sum to 1, all non-negative
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(self.models))]
        
        result = minimize(objective, initial_weights, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x / np.sum(result.x)
    
    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.models)))
        
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
        
        # Weighted average
        return np.dot(predictions, self.optimized_weights)
```

## Advanced Voting Techniques

### 1. Dynamic Voting

Different models are used for different input regions:

```python
class DynamicVotingEnsemble:
    def __init__(self, models, selector_model):
        self.models = models
        self.selector_model = selector_model  # Decides which models to use
        self.model_expertise = {}
    
    def fit(self, X, y):
        # Train all models
        for model in self.models:
            model.fit(X, y)
        
        # Train selector model to predict best model for each input
        self._learn_model_expertise(X, y)
        
        return self
    
    def _learn_model_expertise(self, X, y):
        """Learn which models work best in which regions"""
        # For each sample, find which model predicts best
        best_models = []
        
        for i in range(X.shape[0]):
            sample_errors = []
            
            for model in self.models:
                # Cross-validation to avoid overfitting
                pred = self._cross_val_predict(model, X, y, i)
                error = abs(pred - y[i])
                sample_errors.append(error)
            
            best_model_idx = np.argmin(sample_errors)
            best_models.append(best_model_idx)
        
        # Train selector to predict best model index
        self.selector_model.fit(X, best_models)
    
    def predict(self, X):
        # Predict which model to use for each sample
        selected_models = self.selector_model.predict(X)
        
        predictions = []
        for i, model_idx in enumerate(selected_models):
            model = self.models[model_idx]
            pred = model.predict(X[i:i+1])
            predictions.append(pred[0])
        
        return np.array(predictions)
```

### 2. Hierarchical Voting

Models are organized in a hierarchy:

```python
class HierarchicalVotingEnsemble:
    def __init__(self, model_groups):
        """
        model_groups: List of lists, each sublist contains models for that level
        """
        self.model_groups = model_groups
        self.n_levels = len(model_groups)
    
    def fit(self, X, y):
        # Train all models at all levels
        for level in self.model_groups:
            for model in level:
                model.fit(X, y)
        return self
    
    def predict(self, X):
        current_predictions = None
        
        for level_idx, level_models in enumerate(self.model_groups):
            if level_idx == 0:
                # First level: regular voting
                level_predictions = []
                for model in level_models:
                    pred = model.predict(X)
                    level_predictions.append(pred)
                current_predictions = np.mean(level_predictions, axis=0)
            else:
                # Higher levels: combine with previous level
                level_predictions = []
                for model in level_models:
                    pred = model.predict(X)
                    level_predictions.append(pred)
                
                level_avg = np.mean(level_predictions, axis=0)
                
                # Weighted combination with previous level
                weight = 0.5 ** level_idx  # Exponential decay
                current_predictions = (1 - weight) * current_predictions + weight * level_avg
        
        return current_predictions
```

### 3. Bayesian Voting

Use Bayesian inference to weight models:

```python
class BayesianVotingEnsemble:
    def __init__(self, models, prior_weights=None):
        self.models = models
        self.prior_weights = prior_weights or np.ones(len(models)) / len(models)
        self.posterior_weights = None
    
    def fit(self, X, y):
        # Train all models
        for model in self.models:
            model.fit(X, y)
        
        # Calculate posterior weights using validation performance
        self.posterior_weights = self._calculate_posterior_weights(X, y)
        
        return self
    
    def _calculate_posterior_weights(self, X, y):
        """Calculate Bayesian posterior weights"""
        # Use cross-validation to get unbiased performance estimates
        from sklearn.model_selection import cross_val_score
        
        likelihoods = []
        for model in self.models:
            # Calculate likelihood as exp(-cross_val_error)
            scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
            avg_score = np.mean(scores)
            likelihood = np.exp(avg_score)  # Convert to likelihood
            likelihoods.append(likelihood)
        
        likelihoods = np.array(likelihoods)
        
        # Posterior = likelihood × prior
        posterior = likelihoods * self.prior_weights
        
        # Normalize to get probabilities
        posterior = posterior / np.sum(posterior)
        
        return posterior
    
    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.models)))
        
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
        
        # Weighted average using posterior weights
        return np.dot(predictions, self.posterior_weights)
```

---

# PART II: STACKING (STACKED GENERALIZATION)

## Conceptual Foundation

**Stacking** goes beyond simple voting by learning **how to combine** model predictions. Instead of using fixed rules (like averaging), it trains a **meta-learner** to discover the optimal combination strategy.

### Key Insight
Different models have different strengths and weaknesses. A meta-learner can:
- Learn **when** to trust each model
- Discover **non-linear combinations** 
- Adapt to **input-dependent** model reliability

### Architecture

```
Input Features (X) 
     ↓
┌─────────────────────────────────────────────────────┐
│                 Level 0 (Base Models)               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │ Random      │  │ Linear      │  │ SVM         │  │
│  │ Forest      │  │ Regression  │  │             │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────┘
     ↓              ↓              ↓
┌─────────────────────────────────────────────────────┐
│                 Level 1 (Meta-Model)                │
│  ┌─────────────────────────────────────────────────┐│
│  │ Learns optimal combination of base predictions │││
│  │ Input: [pred1, pred2, pred3]                   │││
│  │ Output: Final prediction                       │││
│  └─────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────┘
```

## Stacking Implementation

### 1. Basic Stacking

```python
class BasicStacking:
    def __init__(self, base_models, meta_model, cv_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv_folds = cv_folds
        self.trained_base_models = []
        self.trained_meta_model = None
    
    def fit(self, X, y):
        # Step 1: Create meta-features using cross-validation
        meta_features = self._create_meta_features(X, y)
        
        # Step 2: Train base models on full training set
        self.trained_base_models = []
        for model in self.base_models:
            trained_model = clone(model)
            trained_model.fit(X, y)
            self.trained_base_models.append(trained_model)
        
        # Step 3: Train meta-model on meta-features
        self.trained_meta_model = clone(self.meta_model)
        self.trained_meta_model.fit(meta_features, y)
        
        return self
    
    def _create_meta_features(self, X, y):
        """Create meta-features using cross-validation"""
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        
        # Initialize meta-features array
        meta_features = np.zeros((n_samples, n_models))
        
        # Cross-validation to create meta-features
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X):
            # Split data
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold = y[train_idx]
            
            # Train each base model on training fold
            for i, model in enumerate(self.base_models):
                model_copy = clone(model)
                model_copy.fit(X_train_fold, y_train_fold)
                
                # Predict on validation fold
                pred = model_copy.predict(X_val_fold)
                meta_features[val_idx, i] = pred
        
        return meta_features
    
    def predict(self, X):
        # Get predictions from all base models
        base_predictions = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, model in enumerate(self.trained_base_models):
            base_predictions[:, i] = model.predict(X)
        
        # Meta-model makes final prediction
        return self.trained_meta_model.predict(base_predictions)
```

### 2. Advanced Stacking with Feature Augmentation

```python
class AdvancedStacking:
    def __init__(self, base_models, meta_model, cv_folds=5, 
                 use_original_features=True, use_probabilities=True):
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv_folds = cv_folds
        self.use_original_features = use_original_features
        self.use_probabilities = use_probabilities
        self.feature_scaler = StandardScaler()
        self.trained_base_models = []
        self.trained_meta_model = None
        self.is_classification = None
    
    def fit(self, X, y):
        # Determine if classification or regression
        self.is_classification = len(np.unique(y)) < 20  # Heuristic
        
        # Create comprehensive meta-features
        meta_features = self._create_advanced_meta_features(X, y)
        
        # Train base models on full training set
        self.trained_base_models = []
        for model in self.base_models:
            trained_model = clone(model)
            trained_model.fit(X, y)
            self.trained_base_models.append(trained_model)
        
        # Scale meta-features
        scaled_meta_features = self.feature_scaler.fit_transform(meta_features)
        
        # Train meta-model
        self.trained_meta_model = clone(self.meta_model)
        self.trained_meta_model.fit(scaled_meta_features, y)
        
        return self
    
    def _create_advanced_meta_features(self, X, y):
        """Create comprehensive meta-features"""
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        
        # Calculate feature dimensions
        feature_dims = []
        
        # Base predictions
        feature_dims.append(n_models)
        
        # Probabilities (if classification)
        if self.is_classification and self.use_probabilities:
            n_classes = len(np.unique(y))
            feature_dims.append(n_models * n_classes)
        
        # Model confidence scores
        feature_dims.append(n_models)
        
        # Original features (if requested)
        if self.use_original_features:
            feature_dims.append(X.shape[1])
        
        # Feature interactions
        feature_dims.append(n_models * (n_models - 1) // 2)
        
        total_features = sum(feature_dims)
        meta_features = np.zeros((n_samples, total_features))
        
        # Cross-validation to create meta-features
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold = y[train_idx]
            
            fold_features = []
            
            # Train models and collect features
            trained_fold_models = []
            for model in self.base_models:
                model_copy = clone(model)
                model_copy.fit(X_train_fold, y_train_fold)
                trained_fold_models.append(model_copy)
            
            # Extract various types of features
            fold_features = self._extract_fold_features(
                trained_fold_models, X_val_fold, y_train_fold
            )
            
            meta_features[val_idx, :] = fold_features
        
        return meta_features
    
    def _extract_fold_features(self, models, X_val, y_train):
        """Extract comprehensive features from models"""
        features = []
        
        # 1. Base predictions
        base_preds = []
        for model in models:
            pred = model.predict(X_val)
            base_preds.append(pred)
        
        base_preds = np.array(base_preds).T
        features.append(base_preds)
        
        # 2. Probabilities (if classification)
        if self.is_classification and self.use_probabilities:
            proba_features = []
            for model in models:
                if hasattr(model, 'predict_proba'):
                    probas = model.predict_proba(X_val)
                    proba_features.append(probas)
                else:
                    # Create dummy probabilities
                    preds = model.predict(X_val)
                    dummy_probas = np.zeros((len(preds), len(np.unique(y_train))))
                    for i, pred in enumerate(preds):
                        dummy_probas[i, int(pred)] = 1.0
                    proba_features.append(dummy_probas)
            
            proba_features = np.concatenate(proba_features, axis=1)
            features.append(proba_features)
        
        # 3. Model confidence scores
        confidence_scores = []
        for model in models:
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(X_val)
                confidence = np.max(probas, axis=1)
            else:
                # Use distance from decision boundary or similar
                preds = model.predict(X_val)
                confidence = np.ones(len(preds))  # Placeholder
            
            confidence_scores.append(confidence)
        
        confidence_scores = np.array(confidence_scores).T
        features.append(confidence_scores)
        
        # 4. Original features (if requested)
        if self.use_original_features:
            features.append(X_val)
        
        # 5. Feature interactions (model prediction interactions)
        interactions = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                interaction = base_preds[:, i] * base_preds[:, j]
                interactions.append(interaction)
        
        if interactions:
            interactions = np.array(interactions).T
            features.append(interactions)
        
        # Concatenate all features
        return np.concatenate(features, axis=1)
```

### 3. Multi-Level Stacking

```python
class MultiLevelStacking:
    def __init__(self, level_models, final_meta_model, cv_folds=5):
        """
        level_models: List of lists, each sublist contains models for that level
        final_meta_model: Final meta-model that combines all levels
        """
        self.level_models = level_models
        self.final_meta_model = final_meta_model
        self.cv_folds = cv_folds
        self.trained_models = []
        self.trained_final_model = None
    
    def fit(self, X, y):
        current_features = X.copy()
        
        # Train each level
        for level_idx, level_models in enumerate(self.level_models):
            print(f"Training level {level_idx + 1} with {len(level_models)} models")
            
            # Create meta-features for this level
            level_meta_features = self._create_level_meta_features(
                current_features, y, level_models
            )
            
            # Train models on full dataset
            trained_level_models = []
            for model in level_models:
                trained_model = clone(model)
                trained_model.fit(current_features, y)
                trained_level_models.append(trained_model)
            
            self.trained_models.append(trained_level_models)
            
            # Prepare features for next level
            if level_idx < len(self.level_models) - 1:
                # Combine original features with meta-features
                current_features = np.concatenate([
                    current_features, level_meta_features
                ], axis=1)
        
        # Train final meta-model
        final_meta_features = self._create_final_meta_features(X, y)
        self.trained_final_model = clone(self.final_meta_model)
        self.trained_final_model.fit(final_meta_features, y)
        
        return self
    
    def _create_level_meta_features(self, X, y, level_models):
        """Create meta-features for a specific level"""
        n_samples = X.shape[0]
        n_models = len(level_models)
        meta_features = np.zeros((n_samples, n_models))
        
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold = y[train_idx]
            
            for i, model in enumerate(level_models):
                model_copy = clone(model)
                model_copy.fit(X_train_fold, y_train_fold)
                pred = model_copy.predict(X_val_fold)
                meta_features[val_idx, i] = pred
        
        return meta_features
    
    def _create_final_meta_features(self, X, y):
        """Create features for final meta-model from all levels"""
        all_features = []
        current_features = X.copy()
        
        for level_idx, level_models in enumerate(self.level_models):
            # Get meta-features for this level
            level_meta_features = self._create_level_meta_features(
                current_features, y, level_models
            )
            all_features.append(level_meta_features)
            
            # Update current features for next level
            current_features = np.concatenate([
                current_features, level_meta_features
            ], axis=1)
        
        # Combine all meta-features
        return np.concatenate(all_features, axis=1)
    
    def predict(self, X):
        all_predictions = []
        current_features = X.copy()
        
        # Get predictions from each level
        for level_idx, trained_level_models in enumerate(self.trained_models):
            level_predictions = []
            
            for model in trained_level_models:
                pred = model.predict(current_features)
                level_predictions.append(pred)
            
            level_predictions = np.array(level_predictions).T
            all_predictions.append(level_predictions)
            
            # Update features for next level
            current_features = np.concatenate([
                current_features, level_predictions
            ], axis=1)
        
        # Combine all level predictions
        final_meta_features = np.concatenate(all_predictions, axis=1)
        
        # Final prediction
        return self.trained_final_model.predict(final_meta_features)
```

## Advanced Stacking Techniques

### 1. Adversarial Stacking

```python
class AdversarialStacking:
    def __init__(self, base_models, meta_model, adversarial_model, 
                 cv_folds=5, adversarial_strength=0.1):
        self.base_models = base_models
        self.meta_model = meta_model
        self.adversarial_model = adversarial_model
        self.cv_folds = cv_folds
        self.adversarial_strength = adversarial_strength
    
    def fit(self, X, y):
        # Create meta-features
        meta_features = self._create_meta_features(X, y)
        
        # Train adversarial model to predict meta-features
        self.adversarial_model.fit(X, meta_features)
        
        # Create adversarial meta-features
        adversarial_meta_features = self.adversarial_model.predict(X)
        
        # Combine original and adversarial meta-features
        combined_features = np.concatenate([
            meta_features,
            self.adversarial_strength * adversarial_meta_features
        ], axis=1)
        
        # Train meta-model
        self.meta_model.fit(combined_features, y)
        
        return self
```

### 2. Uncertainty-Aware Stacking

```python
class UncertaintyAwareStacking:
    def __init__(self, base_models, meta_model, cv_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv_folds = cv_folds
        self.uncertainty_models = []
    
    def fit(self, X, y):
        # Create meta-features with uncertainty
        meta_features, uncertainty_features = self._create_uncertain_meta_features(X, y)
        
        # Combine predictions and uncertainties
        combined_features = np.concatenate([meta_features, uncertainty_features], axis=1)
        
        # Train meta-model
        self.meta_model.fit(combined_features, y)
        
        return self
    
    def _create_uncertain_meta_features(self, X, y):
        """Create meta-features with uncertainty estimates"""
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        
        meta_features = np.zeros((n_samples, n_models))
        uncertainty_features = np.zeros((n_samples, n_models))
        
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold = y[train_idx]
            
            for i, model in enumerate(self.base_models):
                # Train model with uncertainty estimation
                if hasattr(model, 'fit_with_uncertainty'):
                    model.fit_with_uncertainty(X_train_fold, y_train_fold)
                    pred, uncertainty = model.predict_with_uncertainty(X_val_fold)
                else:
                    # Use bootstrap for uncertainty estimation
                    pred, uncertainty = self._bootstrap_predict(
                        model, X_train_fold, y_train_fold, X_val_fold
                    )
                
                meta_features[val_idx, i] = pred
                uncertainty_features[val_idx, i] = uncertainty
        
        return meta_features, uncertainty_features
    
    def _bootstrap_predict(self, model, X_train, y_train, X_val, n_bootstrap=10):
        """Estimate uncertainty using bootstrap"""
        predictions = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(X_train), len(X_train), replace=True)
            X_boot = X_train[indices]
            y_boot = y_train[indices]
            
            # Train and predict
            model_copy = clone(model)
            model_copy.fit(X_boot, y_boot)
            pred = model_copy.predict(X_val)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
```

---

# PART III: COMPARATIVE ANALYSIS

## Performance Comparison

### Theoretical Analysis

#### Voting Ensemble
**Strengths**:
- Simple and interpretable
- Computationally efficient
- Robust to individual model failures
- Works well when models have similar performance

**Weaknesses**:
- Cannot learn complex combinations
- Treats all models equally (in basic version)
- Limited adaptability to input patterns

#### Stacking
**Strengths**:
- Learns optimal combination strategy
- Can discover complex, non-linear relationships
- Adapts to input-dependent model reliability
- Generally achieves better performance

**Weaknesses**:
- More complex to implement and tune
- Higher computational cost
- Risk of overfitting (especially with small datasets)
- Less interpretable

### Empirical Comparison

```python
def compare_ensemble_methods(X, y, base_models, test_size=0.2, cv_folds=5):
    """Compare voting vs stacking empirically"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    results = {}
    
    # Individual model performance
    individual_scores = []
    for i, model in enumerate(base_models):
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        individual_scores.append(score)
        results[f'Model_{i}'] = score
    
    # Hard voting
    hard_voting = HardVotingClassifier(base_models)
    hard_voting.fit(X_train, y_train)
    results['Hard_Voting'] = hard_voting.score(X_test, y_test)
    
    # Soft voting
    soft_voting = SoftVotingClassifier(base_models)
    soft_voting.fit(X_train, y_train)
    results['Soft_Voting'] = soft_voting.score(X_test, y_test)
    
    # Basic stacking
    basic_stacking = BasicStacking(base_models, LinearRegression(), cv_folds)
    basic_stacking.fit(X_train, y_train)
    results['Basic_Stacking'] = basic_stacking.score(X_test, y_test)
    
    # Advanced stacking
    advanced_stacking = AdvancedStacking(base_models, RandomForestRegressor(), cv_folds)
    advanced_stacking.fit(X_train, y_train)
    results['Advanced_Stacking'] = advanced_stacking.score(X_test, y_test)
    
    # Statistical significance testing
    significance_tests = {}
    for method in ['Hard_Voting', 'Soft_Voting', 'Basic_Stacking', 'Advanced_Stacking']:
        # Compare with best individual model
        best_individual = max(individual_scores)
        improvement = results[method] - best_individual
        significance_tests[method] = {
            'improvement': improvement,
            'relative_improvement': improvement / best_individual * 100
        }
    
    return results, significance_tests
```

## When to Use Which Method

### Use Voting Ensemble When:

1. **Simple, interpretable solution needed**
2. **Models have similar performance**
3. **Computational resources are limited**
4. **Quick prototyping or baseline needed**
5. **High reliability required** (less prone to overfitting)

### Use Stacking When:

1. **Maximum performance is priority**
2. **Models have diverse strengths/weaknesses**
3. **Sufficient training data available**
4. **Complex patterns in model relationships**
5. **Computational resources allow for training overhead**

## Hybrid Approaches

### Stacked Voting

```python
class StackedVoting:
    def __init__(self, base_models, voting_groups, meta_model, cv_folds=5):
        """
        Combine voting within groups, then stack between groups
        """
        self.base_models = base_models
        self.voting_groups = voting_groups  # List of model indices for each group
        self.meta_model = meta_model
        self.cv_folds = cv_folds
    
    def fit(self, X, y):
        # Create voting ensembles for each group
        self.voting_ensembles = []
        for group_indices in self.voting_groups:
            group_models = [self.base_models[i] for i in group_indices]
            voting_ensemble = SoftVotingClassifier(group_models)
            voting_ensemble.fit(X, y)
            self.voting_ensembles.append(voting_ensemble)
        
        # Create meta-features from voting ensembles
        meta_features = self._create_voting_meta_features(X, y)
        
        # Train meta-model
        self.meta_model.fit(meta_features, y)
        
        return self
    
    def _create_voting_meta_features(self, X, y):
        """Create meta-features from voting ensembles"""
        n_samples = X.shape[0]
        n_groups = len(self.voting_groups)
        meta_features = np.zeros((n_samples, n_groups))
        
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold = y[train_idx]
            
            for i, group_indices in enumerate(self.voting_groups):
                group_models = [self.base_models[j] for j in group_indices]
                voting_ensemble = SoftVotingClassifier(group_models)
                voting_ensemble.fit(X_train_fold, y_train_fold)
                pred = voting_ensemble.predict(X_val_fold)
                meta_features[val_idx, i] = pred
        
        return meta_features
```

### Dynamic Ensemble Selection

```python
class DynamicEnsembleSelection:
    def __init__(self, base_models, selection_strategy='best_local'):
        self.base_models = base_models
        self.selection_strategy = selection_strategy
        self.trained_models = []
        self.competence_regions = {}
    
    def fit(self, X, y):
        # Train all base models
        for model in self.base_models:
            trained_model = clone(model)
            trained_model.fit(X, y)
            self.trained_models.append(trained_model)
        
        # Learn competence regions
        self._learn_competence_regions(X, y)
        
        return self
    
    def _learn_competence_regions(self, X, y):
        """Learn where each model performs best"""
        from sklearn.neighbors import NearestNeighbors
        
        # For each training sample, find which model predicts best
        best_models = []
        
        for i in range(X.shape[0]):
            sample_errors = []
            
            for model in self.trained_models:
                # Use cross-validation to avoid overfitting
                pred = self._cross_val_predict_single(model, X, y, i)
                error = abs(pred - y[i])
                sample_errors.append(error)
            
            best_model_idx = np.argmin(sample_errors)
            best_models.append(best_model_idx)
        
        # Create competence regions using KNN
        self.competence_knn = NearestNeighbors(n_neighbors=5)
        self.competence_knn.fit(X)
        self.competence_labels = best_models
    
    def predict(self, X):
        predictions = []
        
        for i, x in enumerate(X):
            # Find nearest neighbors
            distances, indices = self.competence_knn.kneighbors([x])
            
            # Get competence labels for neighbors
            neighbor_labels = [self.competence_labels[idx] for idx in indices[0]]
            
            # Select most competent model
            most_competent = max(set(neighbor_labels), key=neighbor_labels.count)
            
            # Make prediction
            pred = self.trained_models[most_competent].predict([x])[0]
            predictions.append(pred)
        
        return np.array(predictions)
```

## Conclusion

Both **Voting Ensemble** and **Stacking** are powerful techniques that serve different purposes:

**Voting Ensemble** is the **democratic approach** - simple, reliable, and interpretable. It works on the principle that the collective wisdom of multiple models is better than any individual model.

**Stacking** is the **learned approach** - it uses machine learning to discover the optimal way to combine model predictions. This added complexity typically results in better performance but requires more care to avoid overfitting.

In practice, many successful systems use **hybrid approaches** that combine the simplicity of voting with the adaptability of stacking, creating robust and high-performing ensemble systems.

The choice between them depends on your specific requirements for performance, interpretability, computational resources, and system complexity.