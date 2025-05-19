"""
Utility functions for machine learning
"""
import pandas as pd
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, VotingClassifier, VotingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, classification_report, 
    confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score,
    mean_absolute_error, precision_score, recall_score, f1_score
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

def train_model(df, target_column, model_type=None):
    """
    Train a machine learning model on the dataset.
    Automatically determines if it's a regression or classification problem.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        target_column (str): Target column name
        model_type (str, optional): Specific model type to train. If None, trains multiple models.
        
    Returns:
        dict: Results of the model training including model comparison and explainability features
    """
    # Check if target exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    # Separate features and target
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    # Process feature types
    # Drop any problematic columns like ID columns or columns with all unique values
    # that would cause issues with modeling
    logger.info(f"Initial feature set: {list(X.columns)}")
    
    # Handle ID columns - check if any column has unique values > 80% of rows
    id_like_cols = []
    for col in X.columns:
        unique_ratio = X[col].nunique() / len(X)
        if unique_ratio > 0.8 and X[col].nunique() > 10:
            id_like_cols.append(col)
            logger.warning(f"Column '{col}' has {unique_ratio:.1%} unique values. Likely an ID column, dropping.")
    
    if id_like_cols:
        X = X.drop(columns=id_like_cols)
        logger.info(f"Dropped potential ID columns: {id_like_cols}")
    
    # Keep track of categorical columns for later processing
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    
    # Force conversion of string columns that look numeric
    for col in list(categorical_cols):  # Create a copy of the list to avoid modification during iteration
        try:
            # Check if all values in the column can be converted to numbers
            test_numeric = pd.to_numeric(X[col], errors='coerce')
            if test_numeric.isna().sum() / len(test_numeric) < 0.1:  # If less than 10% are NaN after conversion
                logger.info(f"Converting column '{col}' from string to numeric")
                X[col] = test_numeric
                categorical_cols.remove(col)
                numeric_cols.append(col)
        except:
            # Keep as categorical if conversion fails
            pass
    
    # If no numeric features after preprocessing, raise error
    if len(numeric_cols) == 0:
        raise ValueError("No numeric features found in the dataset after preprocessing")
    
    # Determine problem type
    unique_count = y.nunique()
    
    # Check if target is already numeric
    if not pd.api.types.is_numeric_dtype(y):
        logger.warning(f"Target column '{target_column}' is not numeric. Treating as regression.")
        problem_type = 'regression'
        # Try to convert to numeric if possible
        try:
            y = pd.to_numeric(y, errors='coerce')
            if y.isna().sum() > 0:
                logger.warning(f"Some values in target column could not be converted to numbers ({y.isna().sum()} NaN values created)")
        except Exception as e:
            logger.error(f"Error converting target to numeric: {str(e)}")
            raise ValueError(f"Target column '{target_column}' must be numeric for machine learning models")
    else:
        # If numeric and less than 10 unique values and they're discrete, treat as classification
        try:
            if unique_count < 10 and all(isinstance(val, (int, np.integer)) or (hasattr(val, 'is_integer') and val.is_integer()) for val in y.dropna().unique()):
                problem_type = 'classification'
                # For classification, convert target to integer
                y = y.astype(int)
            else:
                problem_type = 'regression'
        except (TypeError, ValueError) as e:
            # If comparison fails, default to regression
            logger.warning(f"Error determining problem type: {str(e)}. Defaulting to regression.")
            problem_type = 'regression'
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Only add categorical processing if there are categorical columns
    if categorical_cols:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols)
            ])
    
    # Set up Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define all available models - with reduced resource usage and Neural Network removed
    if problem_type == 'classification':
        available_models = {
            'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42),
            'SVM': SVC(probability=True, kernel='linear', random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
            'Naive Bayes': GaussianNB(),
            'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
            # Neural Network removed due to performance issues causing timeouts
            'Ensemble': VotingClassifier(estimators=[
                ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                ('dt', DecisionTreeClassifier(max_depth=10, random_state=42)), 
                ('lr', LogisticRegression(max_iter=500, random_state=42))
            ], voting='soft')
        }
    else:  # regression
        available_models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42),
            'SVM': SVR(kernel='linear'),
            'KNN': KNeighborsRegressor(n_neighbors=5),
            'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
            'ElasticNet': ElasticNet(random_state=42),
            'Lasso': Lasso(random_state=42),
            'Ridge': Ridge(random_state=42),
            'AdaBoost': AdaBoostRegressor(n_estimators=50, random_state=42),
            # Neural Network removed due to performance issues causing timeouts
            'Ensemble': VotingRegressor(estimators=[
                ('rf', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)),
                ('dt', DecisionTreeRegressor(max_depth=10, random_state=42)),
                ('lr', LinearRegression())
            ])
        }
    
    # If a specific model was requested, filter available models
    if model_type and model_type in available_models:
        models_to_try = {model_type: available_models[model_type]}
    else:
        models_to_try = available_models
    
    # Train and evaluate models
    model_results = {}
    best_score = -float('inf')
    best_model_name = None
    best_model = None
    model_name = "Unknown"  # Default value in case of unexpected errors
    
    # Prepare data for evaluation
    X_preprocessed = preprocessor.fit_transform(X)
    X_train_preprocessed = preprocessor.transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    for name, model in models_to_try.items():
        try:
            # Set the current model name for logging/error handling
            model_name = name
            
            # Create a pipeline with preprocessing and model
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # Cross-validation scores with error handling
            try:
                if problem_type == 'classification':
                    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
                else:
                    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='r2')
            except Exception as e:
                logger.warning(f"Cross-validation failed for {name}: {str(e)}")
                cv_scores = np.array([-1.0])  # Use a placeholder value to indicate failure
            
            try:
                # Fit the model on training data
                pipeline.fit(X_train, y_train)
                
                # Predict on test set
                y_pred = pipeline.predict(X_test)
            except Exception as e:
                logger.error(f"Model training failed for {model_name}: {str(e)}")
                # Skip this model and continue with others
                continue
            
            # Calculate model performance metrics
            if problem_type == 'classification':
                test_score = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                
                # Generate ROC curve if model supports predict_proba
                roc_data = None
                if hasattr(model, 'predict_proba'):
                    try:
                        y_prob = pipeline.predict_proba(X_test)
                        # For multiclass, we'll just use the first class for the ROC
                        if y_prob.shape[1] == 2:  # binary classification
                            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                            roc_auc = auc(fpr, tpr)
                            roc_data = {
                                'fpr': fpr.tolist(),
                                'tpr': tpr.tolist(),
                                'auc': roc_auc
                            }
                    except Exception as e:
                        logger.error(f"Error generating ROC curve for {name}: {str(e)}")
                
                # Calculate precision, recall, f1
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Store model results
                model_results[name] = {
                    'cv_scores': cv_scores.tolist(),
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'test_score': test_score,
                    'confusion_matrix': cm.tolist(),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True),
                    'roc_data': roc_data,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
                
                # Update best model if this one is better
                if cv_scores.mean() > best_score:
                    best_score = cv_scores.mean()
                    best_model_name = name
                    best_model = pipeline
                
            else:  # regression
                test_r2 = r2_score(y_test, y_pred)
                test_mse = mean_squared_error(y_test, y_pred)
                test_mae = mean_absolute_error(y_test, y_pred)
                
                # Calculate residuals
                residuals = y_test - y_pred
                
                # Store model results
                model_results[name] = {
                    'cv_scores': cv_scores.tolist(),
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'test_r2': test_r2,
                    'test_mse': test_mse,
                    'test_mae': test_mae,
                    'residuals': {
                        'mean': residuals.mean(),
                        'std': residuals.std(),
                        'min': residuals.min(),
                        'max': residuals.max()
                    }
                }
                
                # Update best model if this one is better
                if cv_scores.mean() > best_score:
                    best_score = cv_scores.mean()
                    best_model_name = name
                    best_model = pipeline
            
            logger.info(f"{name} CV score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            
        except Exception as e:
            logger.error(f"Error training {name}: {str(e)}")
            model_results[name] = {'error': str(e)}
    
    # Ensure we have at least one successful model
    if best_model is None or best_model_name is None:
        raise ValueError("All models failed to train")
    
    # Get feature importance for the best model
    feature_importance = {}
    feature_names = numeric_cols.copy()
    
    # For models that include one-hot encoding, we need to extract the feature names
    if categorical_cols:
        # Get the categorical transformer
        cat_transformer = preprocessor.named_transformers_['cat']
        # Get the one-hot encoder
        onehot = cat_transformer.named_steps['onehot']
        # Get the encoded feature names
        encoded_features = onehot.get_feature_names_out(categorical_cols)
        # Add to feature names
        feature_names.extend(encoded_features)
    
    # Extract the model from the pipeline
    model = best_model.named_steps['model']
    
    # Try different methods to get feature importance based on the model type
    if hasattr(model, 'feature_importances_'):
        # For tree-based models (Random Forest, Gradient Boosting, etc.)
        importances = model.feature_importances_
        for i, feature in enumerate(feature_names):
            if i < len(importances):  # Safety check
                feature_importance[feature] = float(importances[i])  # Convert to float for JSON serialization
                
    elif hasattr(model, 'coef_'):
        # For linear models (Linear Regression, Logistic Regression, etc.)
        # Handle both single and multi-class cases
        if problem_type == 'classification' and len(model.classes_) > 2:
            # Multi-class case, average across all classes
            importances = np.mean(np.abs(model.coef_), axis=0)
        else:
            # Binary classification or regression
            importances = np.abs(model.coef_[0] if hasattr(model.coef_, 'shape') and len(model.coef_.shape) > 1 else model.coef_)
        
        for i, feature in enumerate(feature_names):
            if i < len(importances):  # Safety check
                feature_importance[feature] = float(importances[i])  # Convert to float for JSON serialization
    
    # If we couldn't get feature importance from the model directly, try permutation importance
    if not feature_importance and hasattr(model, 'predict'):
        try:
            # Calculate permutation importance
            perm_importance = permutation_importance(
                best_model, X_test, y_test, n_repeats=10, random_state=42
            )
            
            for i, feature in enumerate(feature_names):
                if i < len(perm_importance.importances_mean):
                    feature_importance[feature] = float(perm_importance.importances_mean[i])
        except Exception as e:
            logger.error(f"Error calculating permutation importance: {str(e)}")
    
    # Normalize feature importance if we have any
    if feature_importance:
        max_value = max(feature_importance.values())
        feature_importance = {k: v/max_value for k, v in feature_importance.items()}
        
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))
    
    # Generate model explanations and visualizations
    visualizations = {}
    
    # 1. Feature Importance Visualization (top 10 features)
    if feature_importance:
        try:
            # Sort features by importance and get top 10
            top_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
            
            # Create a bar plot
            plt.figure(figsize=(10, 6))
            plt.barh(list(top_features.keys()), list(top_features.values()), color='skyblue')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title('Top 10 Feature Importance')
            plt.tight_layout()
            
            # Save figure to a base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
            feature_importance_img = base64.b64encode(buffer.read()).decode('utf-8')
            visualizations['feature_importance'] = feature_importance_img
        except Exception as e:
            logger.error(f"Error generating feature importance visualization: {str(e)}")
    
    # 2. Model Comparison Visualization
    try:
        # Extract CV scores for each model
        model_names = list(model_results.keys())
        cv_means = [model_results[name].get('cv_mean', 0) if 'error' not in model_results[name] else 0 for name in model_names]
        cv_stds = [model_results[name].get('cv_std', 0) if 'error' not in model_results[name] else 0 for name in model_names]
        
        # Sort by performance
        sorted_indices = np.argsort(cv_means)[::-1]
        model_names = [model_names[i] for i in sorted_indices]
        cv_means = [cv_means[i] for i in sorted_indices]
        cv_stds = [cv_stds[i] for i in sorted_indices]
        
        # Create a bar plot with error bars
        plt.figure(figsize=(12, 8))
        bars = plt.barh(model_names, cv_means, xerr=cv_stds, color='lightgreen', alpha=0.7)
        plt.xlabel('Cross-Validation Score (higher is better)')
        plt.title('Model Comparison')
        plt.tight_layout()
        
        # Add value labels to the bars
        for i, bar in enumerate(bars):
            plt.text(cv_means[i] + 0.01, bar.get_y() + bar.get_height()/2, 
                     f'{cv_means[i]:.3f} (±{cv_stds[i]:.3f})', 
                     va='center')
        
        # Save figure to a base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        model_comparison_img = base64.b64encode(buffer.read()).decode('utf-8')
        visualizations['model_comparison'] = model_comparison_img
    except Exception as e:
        logger.error(f"Error generating model comparison visualization: {str(e)}")
    
    # 3. Problem-specific visualizations
    if problem_type == 'classification':
        try:
            # Confusion Matrix Heatmap
            cm = model_results[best_model_name]['confusion_matrix']
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            
            # Save figure to a base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
            confusion_matrix_img = base64.b64encode(buffer.read()).decode('utf-8')
            visualizations['confusion_matrix'] = confusion_matrix_img
            
            # ROC Curve (if available)
            if 'roc_data' in model_results[best_model_name] and model_results[best_model_name]['roc_data']:
                roc_data = model_results[best_model_name]['roc_data']
                plt.figure(figsize=(8, 6))
                plt.plot(roc_data['fpr'], roc_data['tpr'], color='darkorange', lw=2,
                         label=f'ROC curve (area = {roc_data["auc"]:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")
                plt.tight_layout()
                
                # Save figure to a base64 string
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                plt.close()
                buffer.seek(0)
                roc_curve_img = base64.b64encode(buffer.read()).decode('utf-8')
                visualizations['roc_curve'] = roc_curve_img
                
        except Exception as e:
            logger.error(f"Error generating classification visualizations: {str(e)}")
    else:  # Regression
        try:
            # Actual vs Predicted Plot
            y_pred = best_model.predict(X_test)
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title('Actual vs Predicted')
            plt.tight_layout()
            
            # Save figure to a base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
            actual_vs_predicted_img = base64.b64encode(buffer.read()).decode('utf-8')
            visualizations['actual_vs_predicted'] = actual_vs_predicted_img
            
            # Residuals Plot
            residuals = y_test - y_pred
            plt.figure(figsize=(8, 6))
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.xlabel('Predicted')
            plt.ylabel('Residuals')
            plt.title('Residuals Plot')
            plt.tight_layout()
            
            # Save figure to a base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
            residuals_plot_img = base64.b64encode(buffer.read()).decode('utf-8')
            visualizations['residuals_plot'] = residuals_plot_img
            
        except Exception as e:
            logger.error(f"Error generating regression visualizations: {str(e)}")
    
    # 4. Generate a Decision Tree visualization if applicable
    if best_model_name == 'Decision Tree':
        try:
            model = best_model.named_steps['model']
            plt.figure(figsize=(16, 10))
            plot_tree(model, filled=True, feature_names=feature_names, proportion=True, max_depth=3)
            plt.title('Decision Tree Visualization (limited to depth 3)')
            plt.tight_layout()
            
            # Save figure to a base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            plt.close()
            buffer.seek(0)
            tree_viz_img = base64.b64encode(buffer.read()).decode('utf-8')
            visualizations['decision_tree'] = tree_viz_img
        except Exception as e:
            logger.error(f"Error generating decision tree visualization: {str(e)}")
    
    # Gather dataset statistics
    dataset_stats = {
        'total_records': len(df),
        'training_records': len(X_train),
        'testing_records': len(X_test),
        'numeric_features': len(numeric_cols),
        'categorical_features': len(categorical_cols),
        'total_features': len(numeric_cols) + len(categorical_cols),
        'target_distribution': y.value_counts().to_dict() if problem_type == 'classification' else {
            'min': float(y.min()),
            'max': float(y.max()),
            'mean': float(y.mean()),
            'median': float(y.median())
        }
    }
    
    # Generate model insights and explanations
    model_explainability = {
        'feature_importance': feature_importance,
        'model_comparison': {name: model_results[name] for name in model_names[:5]},  # Top 5 models
        'cross_validation': {
            'strategy': 'K-Fold Cross Validation',
            'folds': 5,
            'scoring_metric': 'accuracy' if problem_type == 'classification' else 'r2'
        },
        'performance': {
            'best_model': best_model_name,
            'best_score': float(best_score)
        },
        'interpretation': get_model_interpretation(best_model_name, problem_type, model_results[best_model_name], feature_importance)
    }
    
    # Return the complete results
    return {
        'problem_type': problem_type,
        'best_model': {
            'name': best_model_name,
            'performance': model_results[best_model_name],
        },
        'dataset_stats': dataset_stats,
        'model_explainability': model_explainability,
        'visualizations': visualizations,
        'all_models': model_results
    }


def get_model_interpretation(model_name, problem_type, performance, feature_importance):
    """Generate a text interpretation of the model results."""
    
    # Get top 3 most important features
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
    top_features_text = ", ".join([f"{feature} ({importance:.2f})" for feature, importance in top_features])
    
    # Basic model description
    model_descriptions = {
        'Logistic Regression': "a linear model for classification that predicts probabilities using the logistic function",
        'Linear Regression': "a linear approach to modeling the relationship between inputs and a continuous outcome",
        'Random Forest': "an ensemble of decision trees that reduces overfitting and improves accuracy",
        'Gradient Boosting': "a method that builds trees one at a time, where each new tree helps to correct errors made by previously trained trees",
        'SVM': "a model that finds a hyperplane that best separates the classes with the maximum margin",
        'KNN': "a non-parametric method that classifies based on the majority vote of its neighbors",
        'Decision Tree': "a flowchart-like structure where each internal node represents a test on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label",
        'Neural Network': "a computational model inspired by the structure of biological neural networks",
        'AdaBoost': "an ensemble method that adjusts subsequent weak learners to focus on instances previously misclassified",
        'Ensemble': "a model that combines the predictions from multiple models to improve overall performance",
        'ElasticNet': "a linear model with combined L1 and L2 regularization penalties",
        'Lasso': "a linear model with L1 regularization that can produce sparse models",
        'Ridge': "a linear model with L2 regularization that reduces overfitting",
        'Naive Bayes': "a probabilistic classifier based on applying Bayes' theorem with strong independence assumptions between features"
    }
    
    model_description = model_descriptions.get(model_name, "a machine learning model")
    
    # Performance interpretation
    if problem_type == 'classification':
        accuracy = performance.get('test_score', 0)
        cv_mean = performance.get('cv_mean', 0)
        
        # Interpret accuracy
        if accuracy > 0.9:
            accuracy_quality = "excellent"
        elif accuracy > 0.8:
            accuracy_quality = "good"
        elif accuracy > 0.7:
            accuracy_quality = "moderate"
        elif accuracy > 0.6:
            accuracy_quality = "fair"
        else:
            accuracy_quality = "poor"
            
        # Check for overfitting
        train_test_diff = cv_mean - accuracy
        overfitting_text = ""
        if abs(train_test_diff) > 0.1:
            if train_test_diff > 0:
                overfitting_text = " There are signs of overfitting, as the model performs better on the training data than on the test data."
            else:
                overfitting_text = " There are signs of underfitting, as the model performs better on the test data than on cross-validation."
                
        performance_text = f"The model achieved {accuracy_quality} accuracy of {accuracy:.2f} on the test data.{overfitting_text}"
        
    else:  # Regression
        r2 = performance.get('test_r2', 0)
        cv_mean = performance.get('cv_mean', 0)
        
        # Interpret R²
        if r2 > 0.8:
            r2_quality = "excellent"
        elif r2 > 0.6:
            r2_quality = "good"
        elif r2 > 0.4:
            r2_quality = "moderate"
        elif r2 > 0.2:
            r2_quality = "fair"
        else:
            r2_quality = "poor"
            
        # Check for overfitting
        train_test_diff = cv_mean - r2
        overfitting_text = ""
        if abs(train_test_diff) > 0.1:
            if train_test_diff > 0:
                overfitting_text = " There are signs of overfitting, as the model performs better on the training data than on the test data."
            else:
                overfitting_text = " There are signs of underfitting, as the model performs better on the test data than on cross-validation."
                
        performance_text = f"The model achieved {r2_quality} R² value of {r2:.2f} on the test data.{overfitting_text}"
    
    # Combine all parts
    interpretation = (
        f"The best model for this {problem_type} problem is {model_name}, which is {model_description}. "
        f"{performance_text} "
        f"The most important features for this model are {top_features_text}."
    )
    
    return interpretation

def time_series_forecast(df, date_column, target_column, forecast_periods=30, method='auto'):
    """
    Perform time series forecasting on a dataset.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        date_column (str): Column name for dates
        target_column (str): Column name for the target variable to forecast
        forecast_periods (int): Number of periods to forecast
        method (str): Forecasting method ('auto', 'arima', 'sarima', 'exponential')
        
    Returns:
        dict: Results of the time series forecasting
    """
    # Validate inputs
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in DataFrame")
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    # Make a copy of the dataframe to avoid modifying the original
    ts_df = df.copy()
    
    # Ensure date column is datetime type
    ts_df[date_column] = pd.to_datetime(ts_df[date_column])
    
    # Sort by date
    ts_df = ts_df.sort_values(date_column)
    
    # Set date as index
    ts_df = ts_df.set_index(date_column)
    
    # Extract the target series
    target_series = ts_df[target_column]
    
    # Check if series has enough data points
    if len(target_series) < 10:
        raise ValueError("Not enough data points for time series forecasting. Need at least 10 data points.")
    
    # Analyze time series characteristics
    ts_analysis = {}
    
    # Check for seasonality (at least 2 years of data needed)
    try:
        if len(target_series) >= 24:  # Assuming monthly data, need at least 2 years
            decomposition = seasonal_decompose(target_series, model='additive')
            seasonal = decomposition.seasonal
            ts_analysis['seasonality_strength'] = abs(seasonal).mean() / abs(decomposition.trend).mean()
            ts_analysis['has_seasonality'] = ts_analysis['seasonality_strength'] > 0.1
        else:
            ts_analysis['has_seasonality'] = False
    except Exception as e:
        logger.warning(f"Could not analyze seasonality: {str(e)}")
        ts_analysis['has_seasonality'] = False
    
    # Determine frequency of data
    try:
        ts_index = ts_df.index
        time_deltas = []
        for i in range(1, min(len(ts_index), 10)):
            delta = ts_index[i] - ts_index[i-1]
            time_deltas.append(delta.days)
        
        avg_delta = np.mean(time_deltas)
        
        if avg_delta < 2:
            frequency = 'D'  # Daily
            seasonal_periods = 7  # Weekly seasonality
        elif avg_delta < 8:
            frequency = 'W'  # Weekly
            seasonal_periods = 52  # Yearly seasonality
        elif avg_delta < 32:
            frequency = 'M'  # Monthly
            seasonal_periods = 12  # Yearly seasonality
        elif avg_delta < 92:
            frequency = 'Q'  # Quarterly
            seasonal_periods = 4  # Yearly seasonality
        else:
            frequency = 'Y'  # Yearly
            seasonal_periods = 1  # No clear seasonality
            
        ts_analysis['frequency'] = frequency
        ts_analysis['seasonal_periods'] = seasonal_periods
    except Exception as e:
        logger.warning(f"Could not determine frequency: {str(e)}")
        ts_analysis['frequency'] = 'Unknown'
        seasonal_periods = 1
    
    # Determine best method if auto
    if method == 'auto':
        if ts_analysis.get('has_seasonality', False):
            method = 'sarima'
        elif len(target_series) > 40:
            method = 'arima'
        else:
            method = 'exponential'
    
    # Create forecasting model based on method
    forecasts = {}
    try:
        if method == 'arima':
            # Simple ARIMA model (p=1, d=1, q=1)
            model = ARIMA(target_series, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Get forecast
            forecast = fitted_model.forecast(steps=forecast_periods)
            forecasts['predictions'] = forecast.values.tolist()
            
            # Calculate forecast dates
            last_date = ts_df.index[-1]
            forecast_dates = []
            
            if ts_analysis['frequency'] == 'D':
                for i in range(1, forecast_periods + 1):
                    forecast_dates.append((last_date + timedelta(days=i)).strftime('%Y-%m-%d'))
            elif ts_analysis['frequency'] == 'W':
                for i in range(1, forecast_periods + 1):
                    forecast_dates.append((last_date + timedelta(weeks=i)).strftime('%Y-%m-%d'))
            elif ts_analysis['frequency'] == 'M':
                for i in range(1, forecast_periods + 1):
                    next_month = last_date.replace(day=1) + pd.DateOffset(months=i)
                    forecast_dates.append(next_month.strftime('%Y-%m-%d'))
            else:
                for i in range(1, forecast_periods + 1):
                    forecast_dates.append(f"Period {i}")
            
            forecasts['dates'] = forecast_dates
            
            # Model details
            forecasts['model_details'] = {
                'method': 'ARIMA',
                'order': '(1,1,1)',
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            }
        
        elif method == 'sarima':
            # Seasonal ARIMA model
            model = SARIMAX(target_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, seasonal_periods))
            fitted_model = model.fit(disp=False)
            
            # Get forecast
            forecast = fitted_model.forecast(steps=forecast_periods)
            forecasts['predictions'] = forecast.values.tolist()
            
            # Calculate forecast dates (same as ARIMA)
            last_date = ts_df.index[-1]
            forecast_dates = []
            
            if ts_analysis['frequency'] == 'D':
                for i in range(1, forecast_periods + 1):
                    forecast_dates.append((last_date + timedelta(days=i)).strftime('%Y-%m-%d'))
            elif ts_analysis['frequency'] == 'W':
                for i in range(1, forecast_periods + 1):
                    forecast_dates.append((last_date + timedelta(weeks=i)).strftime('%Y-%m-%d'))
            elif ts_analysis['frequency'] == 'M':
                for i in range(1, forecast_periods + 1):
                    next_month = last_date.replace(day=1) + pd.DateOffset(months=i)
                    forecast_dates.append(next_month.strftime('%Y-%m-%d'))
            else:
                for i in range(1, forecast_periods + 1):
                    forecast_dates.append(f"Period {i}")
            
            forecasts['dates'] = forecast_dates
            
            # Model details
            forecasts['model_details'] = {
                'method': 'SARIMA',
                'order': '(1,1,1)(1,1,1,s)',
                'seasonal_periods': seasonal_periods,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            }
            
        else:  # exponential
            # Holt-Winters Exponential Smoothing
            model = ExponentialSmoothing(
                target_series, 
                trend='add',
                seasonal='add' if ts_analysis.get('has_seasonality', False) else None, 
                seasonal_periods=seasonal_periods if ts_analysis.get('has_seasonality', False) else None
            )
            fitted_model = model.fit()
            
            # Get forecast
            forecast = fitted_model.forecast(forecast_periods)
            forecasts['predictions'] = forecast.values.tolist()
            
            # Calculate forecast dates (same as others)
            last_date = ts_df.index[-1]
            forecast_dates = []
            
            if ts_analysis['frequency'] == 'D':
                for i in range(1, forecast_periods + 1):
                    forecast_dates.append((last_date + timedelta(days=i)).strftime('%Y-%m-%d'))
            elif ts_analysis['frequency'] == 'W':
                for i in range(1, forecast_periods + 1):
                    forecast_dates.append((last_date + timedelta(weeks=i)).strftime('%Y-%m-%d'))
            elif ts_analysis['frequency'] == 'M':
                for i in range(1, forecast_periods + 1):
                    next_month = last_date.replace(day=1) + pd.DateOffset(months=i)
                    forecast_dates.append(next_month.strftime('%Y-%m-%d'))
            else:
                for i in range(1, forecast_periods + 1):
                    forecast_dates.append(f"Period {i}")
            
            forecasts['dates'] = forecast_dates
            
            # Model details
            forecasts['model_details'] = {
                'method': 'Exponential Smoothing',
                'trend': 'additive',
                'seasonal': 'additive' if ts_analysis.get('has_seasonality', False) else 'none',
                'seasonal_periods': seasonal_periods if ts_analysis.get('has_seasonality', False) else None
            }
    
    except Exception as e:
        logger.error(f"Error in time series forecasting: {str(e)}")
        raise ValueError(f"Failed to forecast time series: {str(e)}")
    
    # Create historical and forecast visualization
    try:
        # Convert historical data to list format
        historical_dates = [d.strftime('%Y-%m-%d') for d in ts_df.index]
        historical_values = target_series.values.tolist()
        
        # Create plotly figure
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=historical_dates,
            y=historical_values,
            mode='lines',
            name='Historical Data',
            line=dict(color='blue')
        ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=forecasts['dates'],
            y=forecasts['predictions'],
            mode='lines',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Time Series Forecast: {target_column}',
            xaxis_title='Date',
            yaxis_title=target_column,
            legend_title='Legend',
            template='plotly_dark'
        )
        
        # Convert to HTML
        visualization = fig.to_html(full_html=False, include_plotlyjs='cdn')
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        visualization = None
    
    # Return results
    return {
        'method': method,
        'ts_analysis': ts_analysis,
        'forecast': forecasts,
        'visualization': visualization,
        'date_column': date_column,
        'target_column': target_column,
        'forecast_periods': forecast_periods,
        'num_samples': len(target_series)
    }

def compare_datasets(df1, df2, name1="Dataset 1", name2="Dataset 2"):
    """
    Compare two datasets and provide detailed analysis of their differences.
    
    Args:
        df1 (pd.DataFrame): First DataFrame
        df2 (pd.DataFrame): Second DataFrame
        name1 (str): Name for the first dataset
        name2 (str): Name for the second dataset
        
    Returns:
        dict: Comparison results between the two datasets
    """
    comparison = {}
    
    # Basic structure comparison
    comparison['structure'] = {
        f"{name1}_shape": df1.shape,
        f"{name2}_shape": df2.shape,
        'row_difference': df1.shape[0] - df2.shape[0],
        'row_difference_pct': (df1.shape[0] - df2.shape[0]) / df1.shape[0] * 100 if df1.shape[0] > 0 else 0
    }
    
    # Columns comparison
    columns1 = set(df1.columns)
    columns2 = set(df2.columns)
    
    comparison['columns'] = {
        'common_columns': list(columns1.intersection(columns2)),
        f"{name1}_only_columns": list(columns1 - columns2),
        f"{name2}_only_columns": list(columns2 - columns1),
        'column_count_difference': len(columns1) - len(columns2)
    }
    
    # Data type comparison for common columns
    common_columns = comparison['columns']['common_columns']
    dtypes_comparison = {}
    
    for col in common_columns:
        dtype1 = str(df1[col].dtype)
        dtype2 = str(df2[col].dtype)
        dtypes_comparison[col] = {
            f"{name1}_dtype": dtype1,
            f"{name2}_dtype": dtype2,
            'same_dtype': dtype1 == dtype2
        }
    
    comparison['data_types'] = dtypes_comparison
    
    # Statistical comparison for common numeric columns
    stat_comparison = {}
    
    for col in common_columns:
        # Only compare numeric columns
        if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
            stats1 = df1[col].describe().to_dict()
            stats2 = df2[col].describe().to_dict()
            
            # Calculate differences
            diff = {}
            diff['mean_diff'] = stats1['mean'] - stats2['mean']
            diff['mean_diff_pct'] = (diff['mean_diff'] / stats1['mean'] * 100) if stats1['mean'] != 0 else 0
            diff['std_diff'] = stats1['std'] - stats2['std']
            diff['min_diff'] = stats1['min'] - stats2['min']
            diff['max_diff'] = stats1['max'] - stats2['max']
            diff['range_diff'] = (stats1['max'] - stats1['min']) - (stats2['max'] - stats2['min'])
            
            stat_comparison[col] = {
                f"{name1}_stats": stats1,
                f"{name2}_stats": stats2,
                'differences': diff
            }
    
    comparison['statistics'] = stat_comparison
    
    # Data distribution visualization for key numeric columns
    visualizations = {}
    
    try:
        for col in common_columns[:5]:  # Limit to first 5 common columns for brevity
            if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
                # Create histogram comparison
                fig = go.Figure()
                
                # Add histogram for df1
                fig.add_trace(go.Histogram(
                    x=df1[col],
                    nbinsx=30,
                    name=name1,
                    opacity=0.7
                ))
                
                # Add histogram for df2
                fig.add_trace(go.Histogram(
                    x=df2[col],
                    nbinsx=30,
                    name=name2,
                    opacity=0.7
                ))
                
                # Update layout
                fig.update_layout(
                    title=f'Distribution Comparison: {col}',
                    xaxis_title=col,
                    yaxis_title='Count',
                    barmode='overlay',
                    template='plotly_dark'
                )
                
                # Convert to HTML
                visualizations[col] = fig.to_html(full_html=False, include_plotlyjs='cdn')
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
    
    comparison['visualizations'] = visualizations
    
    # Generate summary insights
    insights = []
    
    # Size difference insight
    if abs(comparison['structure']['row_difference_pct']) > 10:
        insights.append(f"{name1} is {'larger' if comparison['structure']['row_difference'] > 0 else 'smaller'} than {name2} by {abs(comparison['structure']['row_difference_pct']):.1f}% ({abs(comparison['structure']['row_difference'])} rows).")
    
    # Column differences
    only_columns_1 = f"{name1}_only_columns"
    only_columns_2 = f"{name2}_only_columns"
    
    if len(comparison['columns'][only_columns_1]) > 0:
        insights.append(f"{name1} has {len(comparison['columns'][only_columns_1])} unique columns not in {name2}.")
    
    if len(comparison['columns'][only_columns_2]) > 0:
        insights.append(f"{name2} has {len(comparison['columns'][only_columns_2])} unique columns not in {name1}.")
    
    # Statistical differences
    for col, stats in comparison['statistics'].items():
        if abs(stats['differences']['mean_diff_pct']) > 20:  # If mean differs by more than 20%
            insights.append(f"Column '{col}' has significantly different means: {stats[f'{name1}_stats']['mean']:.2f} in {name1} vs {stats[f'{name2}_stats']['mean']:.2f} in {name2} ({abs(stats['differences']['mean_diff_pct']):.1f}% difference).")
    
    comparison['insights'] = insights
    
    return comparison
