"""
Utility functions for data visualization
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import json
import logging
import uuid

# Set default styling for all matplotlib plots
plt.style.use('dark_background')
sns.set_theme(style="darkgrid")

logger = logging.getLogger(__name__)

def create_bar_chart(df, x_column, y_column):
    """
    Create a bar chart using Matplotlib
    
    Args:
        df (pd.DataFrame): Input DataFrame
        x_column (str): Column name for x-axis
        y_column (str): Column name for y-axis
        
    Returns:
        str: HTML representation of the chart
    """
    try:
        # Create a new figure with a specific size
        plt.figure(figsize=(10, 6))
        
        # If x has too many unique values, aggregate
        if df[x_column].nunique() > 20:
            # For numerical x columns
            if pd.api.types.is_numeric_dtype(df[x_column]):
                df['binned'] = pd.cut(df[x_column], bins=10)
                # Group by bins and calculate mean of y
                agg_df = df.groupby('binned')[y_column].mean().reset_index()
                # Convert bins to strings
                agg_df['binned'] = agg_df['binned'].astype(str)
                # Create bar chart
                ax = sns.barplot(x='binned', y=y_column, data=agg_df)
                plt.title(f'Average {y_column} by {x_column} (binned)')
                plt.xticks(rotation=45)
            else:
                # For categorical x columns, take top 20
                top_values = df[x_column].value_counts().nlargest(20).index
                filtered_df = df[df[x_column].isin(top_values)]
                # Group by x and calculate mean of y
                agg_df = filtered_df.groupby(x_column)[y_column].mean().reset_index()
                # Create bar chart
                ax = sns.barplot(x=x_column, y=y_column, data=agg_df)
                plt.title(f'Average {y_column} by {x_column} (top 20)')
                plt.xticks(rotation=45)
        else:
            # If not too many values, group by x and calculate mean of y
            agg_df = df.groupby(x_column)[y_column].mean().reset_index()
            # Create bar chart
            ax = sns.barplot(x=x_column, y=y_column, data=agg_df)
            plt.title(f'Average {y_column} by {x_column}')
            plt.xticks(rotation=45)
        
        # Set axis labels
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        
        # Tight layout
        plt.tight_layout()
        
        # Save the figure to a BytesIO object
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        
        # Encode the image to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Close the figure to free up memory
        plt.close()
        
        # Return HTML image tag with the base64-encoded image
        return f'<img src="data:image/png;base64,{image_base64}" alt="Bar Chart" style="max-width:100%; height:auto;">'
        
    except Exception as e:
        logger.error(f"Error creating bar chart: {str(e)}")
        raise ValueError(f"Error creating bar chart: {str(e)}")

def create_scatter_plot(df, x_column, y_column):
    """
    Create a scatter plot using Matplotlib
    
    Args:
        df (pd.DataFrame): Input DataFrame
        x_column (str): Column name for x-axis
        y_column (str): Column name for y-axis
        
    Returns:
        str: HTML representation of the chart
    """
    try:
        # Create a new figure with a specific size
        plt.figure(figsize=(10, 6))
        
        # If too many points, sample
        if len(df) > 1000:
            df_sample = df.sample(1000, random_state=42)
        else:
            df_sample = df
            
        # Create scatter plot
        ax = sns.scatterplot(x=x_column, y=y_column, data=df_sample, alpha=0.7)
        
        # Add trend line and correlation text if both columns are numeric
        if pd.api.types.is_numeric_dtype(df[x_column]) and pd.api.types.is_numeric_dtype(df[y_column]):
            # Add trend line
            sns.regplot(x=x_column, y=y_column, data=df_sample, 
                      scatter=False, line_kws={"color": "red"})
            
            # Calculate correlation
            corr = df[[x_column, y_column]].corr().iloc[0, 1]
            plt.text(0.5, 0.95, f'Correlation: {corr:.2f}', 
                   transform=ax.transAxes, ha='center')
        
        # Set title and labels
        plt.title(f'Scatter Plot of {y_column} vs {x_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        
        # Tight layout
        plt.tight_layout()
        
        # Save the figure to a BytesIO object
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        
        # Encode the image to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Close the figure to free up memory
        plt.close()
        
        # Return HTML image tag with the base64-encoded image
        return f'<img src="data:image/png;base64,{image_base64}" alt="Scatter Plot" style="max-width:100%; height:auto;">'
        
    except Exception as e:
        logger.error(f"Error creating scatter plot: {str(e)}")
        raise ValueError(f"Error creating scatter plot: {str(e)}")

def create_heatmap(df, columns=None):
    """
    Create a correlation heatmap using Matplotlib/Seaborn
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to include
        
    Returns:
        str: HTML representation of the heatmap
    """
    try:
        # If columns not specified, use all numeric columns
        if columns is None:
            numeric_df = df.select_dtypes(include=['number'])
        else:
            # Filter to specified columns and drop non-numeric ones
            numeric_df = df[columns].select_dtypes(include=['number'])
        
        # Check if we have enough columns
        if numeric_df.shape[1] < 2:
            raise ValueError("Need at least 2 numeric columns for a correlation heatmap")
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create a larger figure for the heatmap
        plt.figure(figsize=(12, 10))
        
        # Create the heatmap with annotations
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Create mask for upper triangle
        heatmap = sns.heatmap(
            corr_matrix, 
            annot=True,  # Show the correlation values
            fmt=".2f",  # Format to 2 decimal places
            cmap='viridis',  # Use viridis colormap
            mask=mask,  # Only show lower triangle to avoid redundancy
            square=True,  # Make cells square-shaped
            linewidths=.5,  # Add thin lines between cells
            cbar_kws={"shrink": .8}  # Make the colorbar more compact
        )
        
        # Add title
        plt.title('Correlation Heatmap', fontsize=16, pad=20)
        
        # Improve readability by rotating x-axis labels
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout to ensure all labels are visible
        plt.tight_layout()
        
        # Save the figure to a BytesIO object
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        
        # Encode the image to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Close the figure to free up memory
        plt.close()
        
        # Return HTML image tag with the base64-encoded image
        return f'<img src="data:image/png;base64,{image_base64}" alt="Correlation Heatmap" style="max-width:100%; height:auto;">'
        
    except Exception as e:
        logger.error(f"Error creating heatmap: {str(e)}")
        raise ValueError(f"Error creating heatmap: {str(e)}")
