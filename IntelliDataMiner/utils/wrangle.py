"""
Utility functions for data wrangling and cleaning
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def clean_dataset(df, options=None):
    """
    Enhanced dataset cleaning with multiple options for data preparation.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        options (dict, optional): Cleaning options to customize the process:
            - drop_id_columns (bool): Whether to auto-detect and drop ID-like columns
            - handle_outliers (bool): Whether to handle outliers using IQR method
            - normalize_numeric (bool): Whether to normalize numeric columns
            - drop_high_null_threshold (float): Threshold for dropping high-null columns (0.0-1.0)
            - text_cleaning (bool): Whether to clean text columns
            - duplicate_handling (str): How to handle duplicates ('drop', 'mark', or None)
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
        dict: Summary of cleaning operations performed
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Set default options if none provided
    if options is None:
        options = {
            'drop_id_columns': True,
            'handle_outliers': True,
            'normalize_numeric': False,
            'drop_high_null_threshold': 0.5,
            'text_cleaning': True,
            'duplicate_handling': 'drop'
        }
    
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Track changes for reporting
    cleaning_summary = {
        'rows_before': len(df),
        'columns_before': len(df.columns),
        'dropped_columns': [],
        'converted_columns': [],
        'encoded_columns': [],
        'outliers_handled': {},
        'nulls_filled': {},
        'duplicates_removed': 0
    }
    
    # Strip column names and standardize them
    original_columns = df_clean.columns.tolist()
    df_clean.columns = [col.strip().lower().replace(' ', '_') for col in df_clean.columns]
    
    # Create mapping of new column names to original
    column_map = {new: old for new, old in zip(df_clean.columns, original_columns)}
    cleaning_summary['column_names_standardized'] = column_map
    
    # Drop rows with all nulls
    rows_before = len(df_clean)
    df_clean = df_clean.dropna(how='all')
    cleaning_summary['all_null_rows_dropped'] = rows_before - len(df_clean)
    
    # Handle duplicates if requested
    if options.get('duplicate_handling') == 'drop':
        rows_before = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        cleaning_summary['duplicates_removed'] = rows_before - len(df_clean)
    elif options.get('duplicate_handling') == 'mark':
        df_clean['is_duplicate'] = df_clean.duplicated()
        cleaning_summary['duplicates_marked'] = df_clean['is_duplicate'].sum()
    
    # Detect and drop ID-like columns (highly unique values)
    if options.get('drop_id_columns', True):
        id_like_cols = []
        for col in df_clean.columns:
            # Skip if column is already dropped
            if col not in df_clean.columns:
                continue
                
            # Check if it looks like an ID column
            unique_ratio = df_clean[col].nunique() / len(df_clean)
            if (unique_ratio > 0.9 and df_clean[col].nunique() > 10 and 
                (col.endswith('_id') or col.endswith('id') or col.startswith('id') or 'uuid' in col.lower())):
                id_like_cols.append(col)
        
        if id_like_cols:
            df_clean = df_clean.drop(columns=id_like_cols)
            cleaning_summary['dropped_columns'].extend(id_like_cols)
            cleaning_summary['id_columns_dropped'] = id_like_cols
    
    # Convert to appropriate types - with better detection
    for col in df_clean.columns:
        # Skip if column is already dropped
        if col not in df_clean.columns:
            continue
            
        # Check if column looks like a date
        if df_clean[col].dtype == 'object':
            date_sample = df_clean[col].dropna().sample(min(5, len(df_clean[col].dropna()))).tolist()
            looks_like_date = all([
                isinstance(x, str) and (
                    ('/' in x and x.replace('/', '').isdigit()) or 
                    ('-' in x and x.replace('-', '').isdigit()) or
                    any(month in x.lower() for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
                ) for x in date_sample if isinstance(x, str)
            ])
            
            if looks_like_date:
                try:
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                    cleaning_summary['converted_columns'].append(f"{col} (to datetime)")
                    continue
                except:
                    pass
                    
        # Try to convert to numeric
        if df_clean[col].dtype == 'object':
            try:
                numeric_values = pd.to_numeric(df_clean[col], errors='coerce')
                # Only convert if we don't lose too many values
                null_increase = numeric_values.isna().sum() - df_clean[col].isna().sum()
                if null_increase / len(df_clean) < 0.1:  # Less than 10% new nulls
                    df_clean[col] = numeric_values
                    cleaning_summary['converted_columns'].append(f"{col} (to numeric)")
            except:
                pass
    
    # Clean text columns
    if options.get('text_cleaning', True):
        for col in df_clean.select_dtypes(include=['object']).columns:
            # Skip if column is already dropped or very large text fields
            if col not in df_clean.columns or df_clean[col].astype(str).str.len().max() > 200:
                continue
                
            # Clean text: lowercase, strip whitespace, remove special chars
            try:
                df_clean[col] = df_clean[col].astype(str).str.lower().str.strip()
                df_clean[col] = df_clean[col].str.replace(r'[^\w\s]', '', regex=True)
                cleaning_summary['converted_columns'].append(f"{col} (text cleaned)")
            except:
                pass
    
    # Handle outliers in numeric columns
    if options.get('handle_outliers', True):
        for col in df_clean.select_dtypes(include=['number']).columns:
            # Skip if column is already dropped
            if col not in df_clean.columns:
                continue
                
            try:
                # Calculate IQR
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count outliers
                outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
                
                if outliers > 0 and outliers / len(df_clean) < 0.1:  # If outliers are less than 10%
                    # Cap outliers instead of removing them
                    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                    cleaning_summary['outliers_handled'][col] = outliers
            except:
                pass
    
    # Normalize numeric columns if requested
    if options.get('normalize_numeric', False):
        from sklearn.preprocessing import StandardScaler
        
        numeric_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            try:
                scaler = StandardScaler()
                df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])
                for col in numeric_cols:
                    cleaning_summary['converted_columns'].append(f"{col} (normalized)")
            except Exception as e:
                logger.warning(f"Normalization failed: {str(e)}")
    
    # Drop columns with high null percentage
    null_threshold = options.get('drop_high_null_threshold', 0.5)
    null_percent = df_clean.isnull().mean()
    cols_to_drop = null_percent[null_percent > null_threshold].index.tolist()
    
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)
        cleaning_summary['dropped_columns'].extend(cols_to_drop)
        cleaning_summary['high_null_columns_dropped'] = cols_to_drop
    
    # Encode categorical variables
    for col in df_clean.select_dtypes(include=['object']).columns:
        # Skip if column is already dropped
        if col not in df_clean.columns:
            continue
            
        # If the column has fewer than 50% unique values, encode it
        if df_clean[col].nunique() / len(df_clean) < 0.5:
            try:
                # Use Label Encoding for ordinal categories
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                cleaning_summary['encoded_columns'].append(col)
            except:
                # If encoding fails, leave as is
                pass
    
    # Fill remaining nulls
    for col in df_clean.columns:
        # Skip if column is already dropped
        if col not in df_clean.columns:
            continue
            
        null_count = df_clean[col].isnull().sum()
        if null_count > 0:
            if df_clean[col].dtype.kind in 'ifc':  # integer, float, complex
                # Fill numeric columns with median
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                cleaning_summary['nulls_filled'][col] = f"{null_count} (with median)"
            elif pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                # Fill datetime with the most recent date
                df_clean[col] = df_clean[col].fillna(df_clean[col].max())
                cleaning_summary['nulls_filled'][col] = f"{null_count} (with most recent date)"
            else:
                # Fill categorical columns with mode
                if not df_clean[col].mode().empty:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                    cleaning_summary['nulls_filled'][col] = f"{null_count} (with mode)"
                else:
                    df_clean[col] = df_clean[col].fillna("unknown")
                    cleaning_summary['nulls_filled'][col] = f"{null_count} (with 'unknown')"
    
    # Update summary statistics
    cleaning_summary['rows_after'] = len(df_clean)
    cleaning_summary['columns_after'] = len(df_clean.columns)
    cleaning_summary['total_rows_removed'] = cleaning_summary['rows_before'] - cleaning_summary['rows_after']
    cleaning_summary['total_columns_removed'] = cleaning_summary['columns_before'] - cleaning_summary['columns_after']
    
    return df_clean, cleaning_summary

def summarize_dataset(df):
    """
    Create a summary of the dataset for AI analysis
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        str: Text summary of the dataset
    """
    # Basic info
    n_rows, n_cols = df.shape
    dtypes = df.dtypes.astype(str).to_dict()
    
    # Numeric columns summary
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_summary = {}
    
    for col in numeric_cols:
        numeric_summary[col] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std()
        }
    
    # Categorical columns summary
    cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    cat_summary = {}
    
    for col in cat_cols:
        value_counts = df[col].value_counts().to_dict()
        # Limit to top 5 values for brevity
        top_values = {k: v for i, (k, v) in enumerate(value_counts.items()) if i < 5}
        
        cat_summary[col] = {
            'unique_values': df[col].nunique(),
            'top_values': top_values
        }
    
    # Missing values
    missing = df.isnull().sum().to_dict()
    
    # Compile the summary
    summary = f"Dataset Summary:\n"
    summary += f"- Rows: {n_rows}, Columns: {n_cols}\n"
    summary += f"- Column Data Types: {dtypes}\n\n"
    
    summary += f"Numeric Columns ({len(numeric_cols)}):\n"
    for col, stats in numeric_summary.items():
        summary += f"- {col}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}, median={stats['median']:.2f}, std={stats['std']:.2f}\n"
    
    summary += f"\nCategorical Columns ({len(cat_cols)}):\n"
    for col, stats in cat_summary.items():
        summary += f"- {col}: {stats['unique_values']} unique values\n"
        summary += f"  Top values: {stats['top_values']}\n"
    
    summary += f"\nMissing Values:\n"
    for col, count in missing.items():
        if count > 0:
            summary += f"- {col}: {count} missing values ({count/n_rows:.1%})\n"
    
    # Correlations (only if there are numeric columns)
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        # Get top 5 correlations
        corr_unstack = corr.unstack()
        corr_unstack = corr_unstack[corr_unstack < 1.0]  # Remove self-correlations
        top_corr = corr_unstack.sort_values(ascending=False)[:5]
        
        summary += f"\nTop Correlations:\n"
        for (col1, col2), value in top_corr.items():
            summary += f"- {col1} and {col2}: {value:.2f}\n"
    
    return summary
