"""
Utility functions for data mining
"""
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

def apriori_mining(df):
    """
    Perform Apriori algorithm for association rule mining
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        dict: Results of association rule mining
    """
    try:
        # Check if data is appropriate for Apriori
        if df.shape[1] < 2:
            raise ValueError("Need at least 2 columns for association rules")
        
        # Convert all columns to binary (0/1) for Apriori
        df_encoded = pd.DataFrame()
        
        for column in df.columns:
            if df[column].dtype == 'object' or df[column].nunique() < 10:
                # For categorical columns, use one-hot encoding
                dummies = pd.get_dummies(df[column], prefix=column, prefix_sep='=')
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
            else:
                # For numerical columns, convert to binary based on median
                median = df[column].median()
                col_name = f"{column}=>{median}"
                df_encoded[col_name] = (df[column] > median).astype(int)
        
        # Run Apriori algorithm
        # Start with a low min_support and increase if too many results
        min_support = 0.1
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        
        # If too few results, lower the support threshold
        if len(frequent_itemsets) < 5 and min_support > 0.01:
            min_support = 0.01
            frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        
        # If still too few results, try even lower
        if len(frequent_itemsets) < 3 and min_support > 0.005:
            min_support = 0.005
            frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        
        # Generate association rules
        if len(frequent_itemsets) > 0:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
            
            # Convert rules to a more readable format
            rules_list = []
            
            for idx, row in rules.iterrows():
                rule = {
                    'antecedents': list(row['antecedents']),
                    'consequents': list(row['consequents']),
                    'support': row['support'],
                    'confidence': row['confidence'],
                    'lift': row['lift']
                }
                rules_list.append(rule)
            
            # Sort by lift
            rules_list.sort(key=lambda x: x['lift'], reverse=True)
            
            # Limit to top 20 rules
            rules_list = rules_list[:20]
            
            return {
                'rules': rules_list,
                'min_support': min_support,
                'num_frequent_itemsets': len(frequent_itemsets)
            }
        else:
            return {
                'rules': [],
                'min_support': min_support,
                'num_frequent_itemsets': 0,
                'warning': 'No frequent itemsets found. Try with more data or different columns.'
            }
            
    except Exception as e:
        logger.error(f"Error in apriori_mining: {str(e)}")
        raise ValueError(f"Error in association rule mining: {str(e)}")

def kmeans_clustering(df, n_clusters=3):
    """
    Perform KMeans clustering on the dataset
    
    Args:
        df (pd.DataFrame): Input DataFrame
        n_clusters (int): Number of clusters
        
    Returns:
        dict: Results of KMeans clustering
    """
    try:
        # Keep only numeric columns
        df_numeric = df.select_dtypes(include=['number'])
        
        if df_numeric.shape[1] == 0:
            raise ValueError("No numeric columns found for clustering")
        
        if df_numeric.shape[0] < n_clusters:
            raise ValueError(f"Number of samples ({df_numeric.shape[0]}) is less than n_clusters ({n_clusters})")
        
        # Fill NaNs with median
        df_numeric = df_numeric.fillna(df_numeric.median())
        
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_numeric)
        
        # Apply KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Calculate silhouette score
        silhouette = silhouette_score(scaled_data, clusters) if n_clusters > 1 else 0
        
        # Get cluster centers
        centers = kmeans.cluster_centers_
        
        # Transform centers back to original scale
        centers_original = scaler.inverse_transform(centers)
        
        # Count samples in each cluster
        cluster_sizes = pd.Series(clusters).value_counts().to_dict()
        
        # Return results
        return {
            'n_clusters': n_clusters,
            'features': df_numeric.columns.tolist(),
            'cluster_centers': centers_original.tolist(),
            'silhouette_score': silhouette,
            'cluster_sizes': cluster_sizes
        }
        
    except Exception as e:
        logger.error(f"Error in kmeans_clustering: {str(e)}")
        raise ValueError(f"Error in KMeans clustering: {str(e)}")
