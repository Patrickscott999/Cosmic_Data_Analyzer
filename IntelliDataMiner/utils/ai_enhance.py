"""
Utility functions for AI-powered data enhancements:
- Anomaly detection
- Automated insight generation
- Data quality assessment
- Predictive analysis recommendations
"""

import numpy as np
import pandas as pd
from scipy import stats
import openai
import os
import json
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Initialize OpenAI client
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_client = openai.OpenAI(api_key=openai_api_key)

def detect_anomalies(df, method='zscore', threshold=3.0, columns=None):
    """
    Detect anomalies in the dataset using various methods.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        method (str): Method to use ('zscore', 'iqr', 'isolation_forest', 'lof', 'dbscan')
        threshold (float): Threshold for z-score and IQR methods
        columns (list): Columns to check for anomalies (None for all numeric columns)
        
    Returns:
        dict: Anomaly detection results with counts and indices
    """
    if columns is None:
        # Only use numeric columns
        columns = df.select_dtypes(include=np.number).columns.tolist()
    
    if not columns:
        return {"error": "No numeric columns found for anomaly detection"}
    
    # Create a copy of the dataframe with only selected columns
    df_numeric = df[columns].copy()
    
    # Replace inf values with NaN
    df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with NaN values for anomaly detection
    df_clean = df_numeric.dropna()
    
    # If after cleaning we have too few rows, return error
    if len(df_clean) < 10:
        return {"error": "Too few non-null numeric values for anomaly detection"}
    
    anomalies = {}
    
    try:
        if method == 'zscore':
            # Z-score method
            z_scores = stats.zscore(df_clean)
            abs_z_scores = np.abs(z_scores)
            anomaly_mask = (abs_z_scores > threshold).any(axis=1)
            anomaly_indices = np.where(anomaly_mask)[0].tolist()
            
            # Get details for each anomalous point
            anomaly_details = []
            for idx in anomaly_indices:
                row_idx = df_clean.index[idx]
                cols = []
                for col in columns:
                    z = abs_z_scores[idx, columns.index(col)]
                    if z > threshold:
                        cols.append({
                            "column": col,
                            "value": float(df_clean.iloc[idx][col]),
                            "z_score": float(z)
                        })
                anomaly_details.append({
                    "row_index": int(row_idx),
                    "columns": cols
                })
            
            anomalies = {
                "method": "z-score",
                "threshold": threshold,
                "total_anomalies": len(anomaly_indices),
                "anomaly_percentage": (len(anomaly_indices) / len(df_clean)) * 100,
                "anomaly_details": anomaly_details
            }
            
        elif method == 'iqr':
            # IQR method
            anomaly_indices = []
            anomaly_details = []
            
            for i, col in enumerate(columns):
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                col_anomalies = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
                for idx, row in col_anomalies.iterrows():
                    if idx not in [d["row_index"] for d in anomaly_details]:
                        anomaly_details.append({
                            "row_index": int(idx),
                            "columns": []
                        })
                    
                    detail_idx = next(i for i, d in enumerate(anomaly_details) if d["row_index"] == idx)
                    anomaly_details[detail_idx]["columns"].append({
                        "column": col,
                        "value": float(row[col]),
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound)
                    })
            
            # Clean up details - only keep rows with actual anomalies
            anomaly_details = [d for d in anomaly_details if d["columns"]]
            anomaly_indices = [d["row_index"] for d in anomaly_details]
            
            anomalies = {
                "method": "IQR",
                "threshold": threshold,
                "total_anomalies": len(anomaly_indices),
                "anomaly_percentage": (len(anomaly_indices) / len(df_clean)) * 100,
                "anomaly_details": anomaly_details
            }
            
        elif method == 'isolation_forest':
            # Isolation Forest
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df_clean)
            
            model = IsolationForest(contamination=0.05, random_state=42)
            preds = model.fit_predict(scaled_data)
            anomaly_mask = preds == -1
            anomaly_indices = np.where(anomaly_mask)[0].tolist()
            
            # Convert to original indices
            anomaly_indices = [int(df_clean.index[i]) for i in anomaly_indices]
            
            # Calculate anomaly scores
            scores = -model.score_samples(scaled_data)
            score_threshold = np.percentile(scores, 95)  # Top 5% as anomalies
            
            # Get details for each anomaly
            anomaly_details = []
            for i, idx in enumerate(anomaly_indices):
                anomaly_details.append({
                    "row_index": idx,
                    "anomaly_score": float(scores[i]),
                    "score_threshold": float(score_threshold)
                })
            
            anomalies = {
                "method": "Isolation Forest",
                "total_anomalies": len(anomaly_indices),
                "anomaly_percentage": (len(anomaly_indices) / len(df_clean)) * 100,
                "anomaly_details": anomaly_details
            }
            
        elif method == 'lof':
            # Local Outlier Factor
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df_clean)
            
            model = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
            preds = model.fit_predict(scaled_data)
            anomaly_mask = preds == -1
            anomaly_indices = np.where(anomaly_mask)[0].tolist()
            
            # Convert to original indices
            anomaly_indices = [int(df_clean.index[i]) for i in anomaly_indices]
            
            # Get LOF scores (negative means more anomalous)
            lof_scores = -model.negative_outlier_factor_
            
            # Get details for each anomaly
            anomaly_details = []
            for i, idx in enumerate(anomaly_indices):
                orig_idx = df_clean.index[i]
                anomaly_details.append({
                    "row_index": int(orig_idx),
                    "lof_score": float(lof_scores[i])
                })
            
            anomalies = {
                "method": "Local Outlier Factor",
                "total_anomalies": len(anomaly_indices),
                "anomaly_percentage": (len(anomaly_indices) / len(df_clean)) * 100,
                "anomaly_details": anomaly_details
            }
            
        elif method == 'dbscan':
            # DBSCAN clustering
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df_clean)
            
            # Determine eps parameter based on data density
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min(10, len(scaled_data)-1))
            nn.fit(scaled_data)
            distances, _ = nn.kneighbors(scaled_data)
            distances = np.sort(distances[:, -1])
            knee_point = int(0.95 * len(distances))  # Heuristic for knee point
            eps = distances[knee_point]
            
            # Run DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=5)
            clusters = dbscan.fit_predict(scaled_data)
            
            # Identify outliers (cluster label -1)
            anomaly_mask = clusters == -1
            anomaly_indices = np.where(anomaly_mask)[0].tolist()
            
            # Convert to original indices
            anomaly_indices = [int(df_clean.index[i]) for i in anomaly_indices]
            
            anomalies = {
                "method": "DBSCAN",
                "eps": float(eps),
                "total_anomalies": len(anomaly_indices),
                "anomaly_percentage": (len(anomaly_indices) / len(df_clean)) * 100,
                "anomaly_indices": anomaly_indices
            }
            
        else:
            return {"error": f"Unknown anomaly detection method: {method}"}
            
    except Exception as e:
        return {"error": f"Error during anomaly detection: {str(e)}"}
    
    return anomalies


def assess_data_quality(df):
    """
    Assess the quality of the dataset and provide cleaning suggestions.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        dict: Data quality assessment and cleaning recommendations
    """
    try:
        # Get basic statistics
        row_count = len(df)
        col_count = len(df.columns)
        total_cells = row_count * col_count
        
        # Calculate missing values
        missing_cells = df.isna().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
        # Column-specific missing values
        cols_missing = {col: {
            "missing_count": int(df[col].isna().sum()),
            "missing_percentage": float((df[col].isna().sum() / row_count) * 100)
        } for col in df.columns}
        
        # Identify columns with high missing rates
        high_missing_cols = [col for col in df.columns if df[col].isna().sum() / row_count > 0.2]
        
        # Identify numeric columns with potential outliers
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        potential_outlier_cols = []
        
        for col in numeric_cols:
            if df[col].count() > 10:  # Only check if enough data
                try:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    outlier_percentage = (outlier_count / df[col].count()) * 100
                    
                    if outlier_percentage > 5:
                        potential_outlier_cols.append({
                            "column": col,
                            "outlier_count": int(outlier_count),
                            "outlier_percentage": float(outlier_percentage)
                        })
                except Exception:
                    # Skip columns where quantiles can't be calculated
                    pass
        
        # Check data types consistency
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        inconsistent_type_cols = []
        
        for col in categorical_cols:
            # Check if column might actually be numeric
            try:
                # Try to convert to numeric
                numeric_conversion = pd.to_numeric(df[col], errors='coerce')
                # If most values convert successfully, flag as inconsistent
                success_rate = numeric_conversion.notna().sum() / df[col].count() if df[col].count() > 0 else 0
                if success_rate > 0.8:  # 80% convert successfully
                    inconsistent_type_cols.append({
                        "column": col,
                        "likely_type": "numeric",
                        "conversion_success_rate": float(success_rate * 100)
                    })
            except:
                pass
                
            # Check for date columns
            try:
                date_conversion = pd.to_datetime(df[col], errors='coerce')
                success_rate = date_conversion.notna().sum() / df[col].count() if df[col].count() > 0 else 0
                if success_rate > 0.8:  # 80% convert successfully
                    inconsistent_type_cols.append({
                        "column": col,
                        "likely_type": "datetime",
                        "conversion_success_rate": float(success_rate * 100)
                    })
            except:
                pass
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = (duplicate_count / row_count) * 100 if row_count > 0 else 0
        
        # Generate cleaning recommendations
        cleaning_recommendations = []
        
        if missing_percentage > 5:
            cleaning_recommendations.append({
                "issue": "high_missing_values",
                "description": f"Dataset has {missing_percentage:.1f}% missing values overall",
                "recommendation": "Consider imputation strategies for missing data or removing columns with excessive missing values"
            })
            
        for col, stats in cols_missing.items():
            if stats["missing_percentage"] > 30:
                cleaning_recommendations.append({
                    "issue": "high_column_missing_values",
                    "column": col,
                    "description": f"Column '{col}' has {stats['missing_percentage']:.1f}% missing values",
                    "recommendation": f"Consider dropping column '{col}' or using imputation"
                })
            elif stats["missing_percentage"] > 5:
                cleaning_recommendations.append({
                    "issue": "moderate_column_missing_values",
                    "column": col,
                    "description": f"Column '{col}' has {stats['missing_percentage']:.1f}% missing values",
                    "recommendation": f"Use mean/median/mode imputation for column '{col}'"
                })
                
        for outlier_col in potential_outlier_cols:
            cleaning_recommendations.append({
                "issue": "potential_outliers",
                "column": outlier_col["column"],
                "description": f"Column '{outlier_col['column']}' has {outlier_col['outlier_percentage']:.1f}% potential outliers",
                "recommendation": f"Investigate outliers in '{outlier_col['column']}' and consider capping/flooring or transforming"
            })
            
        for type_col in inconsistent_type_cols:
            cleaning_recommendations.append({
                "issue": "inconsistent_data_type",
                "column": type_col["column"],
                "description": f"Column '{type_col['column']}' contains values consistent with {type_col['likely_type']} type ({type_col['conversion_success_rate']:.1f}% convertible)",
                "recommendation": f"Convert '{type_col['column']}' to {type_col['likely_type']} type"
            })
            
        if duplicate_percentage > 0:
            cleaning_recommendations.append({
                "issue": "duplicate_rows",
                "description": f"Dataset contains {duplicate_count} duplicate rows ({duplicate_percentage:.1f}%)",
                "recommendation": "Remove duplicate rows to prevent bias in analysis"
            })
        
        # Generate quality score based on issues found
        base_score = 100
        deductions = 0
        
        # Deduct for missing values
        deductions += min(30, missing_percentage / 2)
        
        # Deduct for outliers
        if potential_outlier_cols:
            avg_outlier_pct = sum(c["outlier_percentage"] for c in potential_outlier_cols) / len(potential_outlier_cols)
            deductions += min(15, avg_outlier_pct / 2)
            
        # Deduct for inconsistent types
        deductions += len(inconsistent_type_cols) * 5
        
        # Deduct for duplicates
        deductions += min(10, duplicate_percentage)
        
        quality_score = max(0, base_score - deductions)
        
        quality_assessment = {
            "overall_statistics": {
                "row_count": row_count,
                "column_count": col_count,
                "missing_values": {
                    "count": int(missing_cells),
                    "percentage": float(missing_percentage)
                },
                "duplicate_rows": {
                    "count": int(duplicate_count),
                    "percentage": float(duplicate_percentage)
                }
            },
            "column_statistics": {
                col: {
                    "missing_count": int(df[col].isna().sum()),
                    "missing_percentage": float((df[col].isna().sum() / row_count) * 100),
                    "unique_values": int(df[col].nunique()),
                    "data_type": str(df[col].dtype)
                } for col in df.columns
            },
            "quality_score": float(quality_score),
            "cleaning_recommendations": cleaning_recommendations
        }
        
        return quality_assessment
        
    except Exception as e:
        return {"error": f"Error during data quality assessment: {str(e)}"}


def generate_automated_insights(df, sample_size=1000):
    """
    Generate automated insights from the dataset using OpenAI
    
    Args:
        df (pd.DataFrame): Input DataFrame
        sample_size (int): Maximum number of rows to sample for analysis
        
    Returns:
        dict: Automated insights about the dataset
    """
    try:
        # Perform basic statistics
        row_count = len(df)
        col_count = len(df.columns)
        
        # Get column types
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Sample the dataset if it's too large
        if row_count > sample_size:
            df_sample = df.sample(sample_size, random_state=42)
        else:
            df_sample = df
            
        # Generate basic statistical summary
        if numeric_cols:
            stats_summary = df_sample[numeric_cols].describe().to_dict()
        else:
            stats_summary = {}
            
        # Generate correlations if enough numeric columns
        correlations = None
        if len(numeric_cols) > 1:
            corr_matrix = df_sample[numeric_cols].corr()
            
            # Get top correlations
            correlations = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    col1 = numeric_cols[i]
                    col2 = numeric_cols[j]
                    corr_value = corr_matrix.iloc[i, j]
                    if not np.isnan(corr_value):
                        correlations.append({
                            "column1": col1,
                            "column2": col2,
                            "correlation": float(corr_value)
                        })
            
            # Sort by absolute correlation value
            correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            
            # Keep only top correlations
            correlations = correlations[:10]
        
        # Get category distributions for categorical columns
        category_distributions = {}
        for col in categorical_cols[:5]:  # Limit to 5 cols to keep response size manageable
            if df_sample[col].nunique() <= 20:  # Only for columns with reasonable number of categories
                value_counts = df_sample[col].value_counts(normalize=True).head(10).to_dict()
                category_distributions[col] = {str(k): float(v) for k, v in value_counts.items()}
        
        # Prepare data summary for OpenAI
        data_summary = {
            "dataset_info": {
                "rows": row_count,
                "columns": col_count,
                "column_names": df.columns.tolist(),
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "datetime_columns": datetime_cols
            },
            "basic_statistics": stats_summary,
            "correlations": correlations,
            "category_distributions": category_distributions
        }
        
        # Convert summary to a string representation for OpenAI
        summary_str = json.dumps(data_summary, indent=2)
        
        # Use OpenAI to generate insights
        prompt = f"""
        Analyze this dataset summary and generate 5-7 key insights. Focus on patterns, relationships, and potential business implications.
        
        Dataset Summary:
        {summary_str}
        
        Generate a clear, concise response in JSON format with the following structure:
        {{
            "insights": [
                {{
                    "title": "Brief, descriptive title of the insight",
                    "description": "Detailed explanation of the insight with specific values and observations",
                    "visualization_suggestion": "Suggestion for best visualization type for this insight",
                    "relevance": "Why this insight matters and potential business implications"
                }},
                ...
            ],
            "overall_summary": "A few sentences summarizing the dataset and its key characteristics",
            "additional_analysis_suggestions": [
                "Suggestion 1 for further analysis",
                "Suggestion 2 for further analysis",
                ...
            ]
        }}

        Make insights specific and data-driven, referring to actual values, patterns and anomalies.
        """
        
        # Check if OpenAI API key is available
        if not openai_api_key:
            return {
                "error": "OpenAI API key not configured. Please set the OPENAI_API_KEY environment variable."
            }
        
        # Request insights from OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
            messages=[
                {"role": "system", "content": "You are a data analysis assistant that provides insights about datasets."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.5,
            max_tokens=2000
        )
        
        # Parse and return the insights
        insights = json.loads(response.choices[0].message.content)
        return insights
        
    except Exception as e:
        return {"error": f"Error generating automated insights: {str(e)}"}


def predict_analysis_recommendations(df):
    """
    Generate predictive analysis recommendations based on dataset characteristics
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        dict: Recommended analyses and models
    """
    try:
        # Analyze dataset characteristics
        row_count = len(df)
        col_count = len(df.columns)
        
        # Get column types
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Calculate various metrics to determine appropriate analyses
        metrics = {
            "row_count": row_count,
            "column_count": col_count,
            "numeric_column_count": len(numeric_cols),
            "categorical_column_count": len(categorical_cols),
            "datetime_column_count": len(datetime_cols),
            "numeric_categorical_ratio": len(numeric_cols) / len(categorical_cols) if len(categorical_cols) > 0 else float('inf'),
            "missing_value_percentage": (df.isna().sum().sum() / (row_count * col_count)) * 100 if row_count * col_count > 0 else 0
        }
        
        # Check for time series characteristics
        has_time_series_potential = len(datetime_cols) > 0
        if not datetime_cols and categorical_cols:
            # Check if any categorical column might be a date
            for col in categorical_cols:
                try:
                    pd.to_datetime(df[col], errors='coerce')
                    conversion_success = pd.to_datetime(df[col], errors='coerce').notna().mean() > 0.8
                    if conversion_success:
                        has_time_series_potential = True
                        break
                except:
                    pass
        
        # Check for text analysis potential
        has_text_analysis_potential = False
        for col in categorical_cols:
            # Check if column has long text values
            if df[col].dtype == 'object' and df[col].notna().any():
                avg_len = df[col].astype(str).str.len().mean()
                if avg_len > 50:  # Average length > 50 chars suggests text data
                    has_text_analysis_potential = True
                    break
        
        # Prepare dataset characteristics for OpenAI
        dataset_characteristics = {
            "metrics": metrics,
            "has_time_series_potential": has_time_series_potential,
            "has_text_analysis_potential": has_text_analysis_potential,
            "column_types": {
                "numeric": numeric_cols,
                "categorical": categorical_cols,
                "datetime": datetime_cols
            }
        }
        
        # Sample a few rows for context
        sample_data = df.head(5).to_dict(orient='records')
        
        # Convert characteristics to a string for OpenAI
        characteristics_str = json.dumps(dataset_characteristics, indent=2)
        sample_data_str = json.dumps(sample_data, indent=2)
        
        # Use OpenAI to generate recommendations
        prompt = f"""
        Based on these dataset characteristics and sample data, recommend appropriate analysis approaches, 
        machine learning models, and visualization techniques.
        
        Dataset Characteristics:
        {characteristics_str}
        
        Sample Data:
        {sample_data_str}
        
        Generate recommendations in JSON format with this structure:
        {{
            "recommended_analyses": [
                {{
                    "analysis_type": "Name of analysis type",
                    "description": "Description of the analysis and why it's appropriate",
                    "implementation_steps": ["Step 1", "Step 2", ...],
                    "potential_insights": "What kind of insights this analysis might reveal",
                    "prerequisite_data_preparation": ["Prep step 1", "Prep step 2", ...]
                }},
                ...
            ],
            "recommended_visualizations": [
                {{
                    "visualization_type": "Name of visualization",
                    "appropriate_for": ["Column1", "Column2", ...],
                    "description": "Description of what this visualization would show"
                }},
                ...
            ],
            "machine_learning_potential": {{
                "suitable_for_ml": true/false,
                "recommended_models": [
                    {{
                        "model_type": "Model name",
                        "appropriate_for": "Type of problem this model addresses",
                        "expected_performance": "Expected performance characteristics",
                        "feature_columns": ["Column1", "Column2", ...],
                        "target_column_suggestions": ["PossibleTargetColumn1", ...]
                    }},
                    ...
                ],
                "feature_engineering_suggestions": ["Suggestion1", "Suggestion2", ...]
            }},
            "data_limitations": ["Limitation1", "Limitation2", ...],
            "additional_data_suggestions": ["Suggestion1", "Suggestion2", ...]
        }}

        Make all recommendations specific to this dataset's characteristics.
        """
        
        # Check if OpenAI API key is available
        if not openai_api_key:
            return {
                "error": "OpenAI API key not configured. Please set the OPENAI_API_KEY environment variable."
            }
        
        # Request recommendations from OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
            messages=[
                {"role": "system", "content": "You are a data science advisor that recommends appropriate analysis techniques."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.5,
            max_tokens=2000
        )
        
        # Parse and return the recommendations
        recommendations = json.loads(response.choices[0].message.content)
        return recommendations
        
    except Exception as e:
        return {"error": f"Error generating analysis recommendations: {str(e)}"}