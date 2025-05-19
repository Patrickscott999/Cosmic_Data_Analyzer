import os
import logging
import pandas as pd
import json
import base64
import io
import pickle
from flask import Flask, render_template, request, jsonify, session, make_response, Response, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_cors import CORS
import uuid
from utils import wrangle, ml, mine, viz, ai, ai_enhance
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from flask_login import current_user, login_required

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Base class for SQLAlchemy models
class Base(DeclarativeBase):
    pass

# Create Flask app
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)  # needed for url_for to generate with https

# Enable CORS with support for credentials
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})

# Set a session secret key - either from environment or a default for development
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key_for_aidata_analyzer_app_sessions")

# Configure the SQLAlchemy database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize the database
db = SQLAlchemy(model_class=Base)
db.init_app(app)

# In-memory storage for datasets (until we completely migrate to DB)
DATASETS = {}

# Initialize DB tables
with app.app_context():
    # Import models here to avoid circular imports
    import models
    from models import User, OAuth, Dataset
    db.create_all()

# Import and register authentication blueprint
from replit_auth import make_replit_blueprint, require_login
app.register_blueprint(make_replit_blueprint(), url_prefix="/auth")

# Helper function to get dataset from DB or memory
def get_dataset(dataset_id, use_current=True):
    """Get dataset from database or memory cache.
    
    Args:
        dataset_id (str): ID of the dataset
        use_current (bool): If True, use current version, else use original
        
    Returns:
        pd.DataFrame: The dataset
    """
    # Check memory cache first
    if dataset_id in DATASETS:
        return DATASETS[dataset_id]['current' if use_current else 'original']
    
    # Not in memory, try database
    try:
        from models import Dataset
        dataset = Dataset.query.get(dataset_id)
        
        if dataset:
            # Check if user owns this dataset or is admin
            if current_user.is_authenticated and (dataset.user_id == current_user.id):
                # Load from DB into memory
                df_bytes = dataset.current_data if use_current else dataset.original_data
                df = pickle.loads(df_bytes)
                
                # Cache in memory
                DATASETS[dataset_id] = {
                    'original': pickle.loads(dataset.original_data),
                    'current': pickle.loads(dataset.current_data),
                    'filename': dataset.filename
                }
                
                return df
    except Exception as e:
        logger.error(f"Error retrieving dataset {dataset_id}: {str(e)}")
    
    return None

@app.route('/')
def index():
    """Render the main page of the application."""
    # Check if a specific dataset is requested
    dataset_id = request.args.get('dataset_id')
    if dataset_id and current_user.is_authenticated:
        # Load dataset into memory if it's not already there
        dataset = get_dataset(dataset_id)
        if dataset is not None:
            # Set current dataset ID in session
            session['current_dataset_id'] = dataset_id
            
    return render_template('index.html', user=current_user)

@app.route('/dashboard')
@require_login
def dashboard():
    """Render the dashboard for authenticated users."""
    # Fetch user's datasets from the database
    from models import Dataset
    user_datasets = Dataset.query.filter_by(user_id=current_user.id).order_by(Dataset.created_at.desc()).all()
    return render_template('dashboard.html', user=current_user, datasets=user_datasets)


@app.route('/delete_dataset/<dataset_id>', methods=['DELETE'])
@require_login
def delete_dataset(dataset_id):
    """Delete a dataset."""
    try:
        from models import Dataset
        dataset = Dataset.query.get(dataset_id)
        
        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404
            
        # Check if user owns this dataset
        if dataset.user_id != current_user.id:
            return jsonify({"error": "Unauthorized"}), 403
            
        # Delete from memory cache if present
        if dataset_id in DATASETS:
            del DATASETS[dataset_id]
            
        # Delete from database
        db.session.delete(dataset)
        db.session.commit()
        
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error deleting dataset {dataset_id}: {str(e)}")
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
@require_login
def upload_file():
    """Handle file upload and convert to pandas DataFrame."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if file:
            if file.filename is None or file.filename == '':
                return jsonify({"error": "File has no name. Please upload CSV, Excel, or JSON."}), 400
                
            filename = secure_filename(file.filename)
            if '.' not in filename:
                return jsonify({"error": "File has no extension. Please upload CSV, Excel, or JSON."}), 400
                
            file_extension = filename.rsplit('.', 1)[1].lower()
            
            # Generate a unique ID for this dataset
            dataset_id = str(uuid.uuid4())
            
            # Store in session for reference
            session['current_dataset_id'] = dataset_id
            
            # Save to temporary file first
            import tempfile
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, filename)
            file.save(temp_path)
            
            # Process file based on extension
            try:
                if file_extension == 'csv':
                    df = pd.read_csv(temp_path)
                elif file_extension in ['xls', 'xlsx']:
                    df = pd.read_excel(temp_path)
                elif file_extension == 'json':
                    df = pd.read_json(temp_path)
                else:
                    # Clean up temp files
                    os.remove(temp_path)
                    os.rmdir(temp_dir)
                    return jsonify({"error": "Unsupported file format. Please upload CSV, Excel, or JSON."}), 400
                
                # Get file content as bytes for database storage
                with open(temp_path, 'rb') as f:
                    file_content = f.read()
                
                # Store serialized dataframe as bytes
                df_bytes = pickle.dumps(df)
                
            finally:
                # Clean up temp files
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
            
            # Store the DataFrame in memory for this session
            DATASETS[dataset_id] = {
                'original': df,
                'current': df,
                'filename': filename
            }
            
            # Store in database for persistence
            from models import Dataset
            dataset = Dataset(
                id=dataset_id,
                user_id=current_user.id,
                filename=filename,
                original_data=df_bytes,
                current_data=df_bytes,
                file_type=file_extension
            )
            db.session.add(dataset)
            db.session.commit()
            
            # Basic dataset info
            info = {
                'dataset_id': dataset_id,
                'filename': filename,
                'rows': df.shape[0],
                'columns': df.shape[1],
                'column_names': df.columns.tolist(),
                'dtypes': {col: str(df[col].dtype) for col in df.columns}
            }
            
            return jsonify(info)
            
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/clean', methods=['POST'])
def clean_data():
    """Clean the dataset by handling nulls, normalizing types, and encoding."""
    try:
        data = request.json
        # Get dataset ID from the request data or session as fallback
        dataset_id = data.get('dataset_id') or session.get('current_dataset_id')
        
        if not dataset_id or dataset_id not in DATASETS:
            return jsonify({"error": "No dataset found. Please upload a file first."}), 400
        
        # Get cleaning options if provided
        cleaning_options = data.get('options', {})
        
        # Get the dataset
        df = DATASETS[dataset_id]['current'].copy()
        
        # Apply cleaning operations with options
        df_clean, cleaning_summary = wrangle.clean_dataset(df, options=cleaning_options)
        
        # Update the dataset
        DATASETS[dataset_id]['current'] = df_clean
        
        # Combine standard summary with detailed cleaning summary
        summary = {
            'rows_before': cleaning_summary['rows_before'],
            'rows_after': cleaning_summary['rows_after'],
            'rows_removed': cleaning_summary['total_rows_removed'],
            'columns_before': cleaning_summary['columns_before'],
            'columns_after': cleaning_summary['columns_after'],
            'columns_removed': cleaning_summary['total_columns_removed'],
            'columns': df_clean.columns.tolist(),
            'dtypes': {col: str(df_clean[col].dtype) for col in df_clean.columns},
            'null_counts': df_clean.isnull().sum().to_dict(),
            'cleaning_details': cleaning_summary
        }
        
        return jsonify(summary)
        
    except Exception as e:
        logger.error(f"Error cleaning data: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/ml', methods=['POST'])
def train_model():
    """Train a ML model on the dataset."""
    try:
        data = request.json
        # Get dataset ID from the request data or session as fallback
        dataset_id = data.get('dataset_id') or session.get('current_dataset_id')
        
        if not dataset_id:
            return jsonify({"error": "No dataset found. Please upload a file first."}), 400
        
        # Get dataset using our improved function for consistent handling
        df = get_dataset(dataset_id)
        
        if df is None:
            return jsonify({"error": "Dataset not found or you don't have access to it"}), 400
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Convert all object columns to string to prevent issues
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = df[col].astype(str)
            except Exception as e:
                logger.warning(f"Couldn't convert column {col} to string: {str(e)}")
        
        # Add detailed information for debugging
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Dataset columns: {list(df.columns)}")
        
        # Examine columns in detail for troubleshooting
        for col in df.columns:
            try:
                dtype = df[col].dtype
                sample = str(df[col].iloc[0]) if not df[col].empty else "N/A"
                logger.info(f"Column '{col}' has dtype {dtype}, sample: {sample[:50]}")
                if df[col].isna().sum() > 0:
                    logger.warning(f"Column '{col}' has {df[col].isna().sum()} missing values")
            except Exception as col_e:
                logger.error(f"Error examining column '{col}': {str(col_e)}")
        
        logger.info(f"Dataset types: {df.dtypes.to_dict()}")
        
        # Get target column from request
        target_column = data.get('target_column')
        
        if not target_column or target_column not in df.columns:
            return jsonify({"error": "Invalid target column"}), 400
        
        logger.info(f"Target column: {target_column}")
        
        # Get optional model type
        model_type = data.get('model_type')  # Can be None
        logger.info(f"Selected model type: {model_type}")
        
        # Validate model type if specified
        if model_type:
            allowed_models = [
                'Logistic Regression', 'Random Forest', 'Gradient Boosting', 'SVM', 'KNN', 
                'Decision Tree', 'Naive Bayes', 'AdaBoost', 'Ensemble',
                'Linear Regression', 'ElasticNet', 'Lasso', 'Ridge'
            ]
            if model_type not in allowed_models:
                logger.error(f"Invalid model type requested: {model_type}")
                return jsonify({"error": f"Invalid model type: {model_type}"}), 400
        
        # Train model with enhanced functionality and timeout protection
        try:
            # Simple timeout mechanism
            from threading import Timer
            from functools import wraps
            
            def timeout_handler():
                raise TimeoutError("Model training timed out after 25 seconds")
                
            # Set a timer for 25 seconds
            timer = Timer(25, timeout_handler)
            timer.start()
            
            try:
                model_results = ml.train_model(df, target_column, model_type)
                timer.cancel()  # Cancel timer if training completes successfully
                return jsonify(model_results)
            finally:
                timer.cancel()  # Ensure timer is cancelled
                
        except TimeoutError:
            logger.error("Model training timed out")
            return jsonify({"error": "Model training timed out. Try a simpler model or a smaller dataset."}), 408
        except ValueError as ve:
            logger.error(f"Value error in model training: {str(ve)}")
            return jsonify({"error": f"Data validation error: {str(ve)}"}), 400
        except Exception as inner_e:
            logger.error(f"Error during model training: {str(inner_e)}")
            return jsonify({"error": f"Model training failed: {str(inner_e)}"}), 500
        
    except Exception as e:
        import traceback
        logger.error(f"Error in train_model route: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/mine', methods=['POST'])
def mine_data():
    """Perform data mining (Apriori or KMeans)."""
    try:
        data = request.json
        # Get dataset ID from the request data or session as fallback
        dataset_id = data.get('dataset_id') or session.get('current_dataset_id')
        
        if not dataset_id or dataset_id not in DATASETS:
            return jsonify({"error": "No dataset found. Please upload a file first."}), 400
        
        # Get the dataset
        df = DATASETS[dataset_id]['current'].copy()
        
        # Get parameters from request
        method = data.get('method', 'apriori')
        columns = data.get('columns', [])
        
        # Validate columns
        for col in columns:
            if col not in df.columns:
                return jsonify({"error": f"Column {col} not found in dataset"}), 400
        
        # If no columns specified, use all
        if not columns:
            columns = df.columns.tolist()
        
        # Mine data
        if method == 'apriori':
            mining_results = mine.apriori_mining(df[columns])
        elif method == 'kmeans':
            n_clusters = data.get('n_clusters', 3)
            mining_results = mine.kmeans_clustering(df[columns], n_clusters)
        else:
            return jsonify({"error": "Invalid mining method. Use 'apriori' or 'kmeans'."}), 400
        
        return jsonify(mining_results)
        
    except Exception as e:
        logger.error(f"Error mining data: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/visualize', methods=['POST'])
def visualize_data():
    """Generate visualizations of the dataset."""
    try:
        data = request.json
        # Get dataset ID from the request data or session as fallback
        dataset_id = data.get('dataset_id') or session.get('current_dataset_id')
        
        if not dataset_id or dataset_id not in DATASETS:
            return jsonify({"error": "No dataset found. Please upload a file first."}), 400
        
        # Get the dataset
        df = DATASETS[dataset_id]['current'].copy()
        
        # Get parameters from request
        vis_type = data.get('type', 'bar')
        x_column = data.get('x')
        y_column = data.get('y')
        
        # Validate columns
        if not x_column or x_column not in df.columns:
            return jsonify({"error": "Invalid x column"}), 400
        
        if vis_type != 'heatmap' and (not y_column or y_column not in df.columns):
            return jsonify({"error": "Invalid y column"}), 400
        
        # Generate visualization
        if vis_type == 'bar':
            fig_html = viz.create_bar_chart(df, x_column, y_column)
        elif vis_type == 'scatter':
            fig_html = viz.create_scatter_plot(df, x_column, y_column)
        elif vis_type == 'heatmap':
            fig_html = viz.create_heatmap(df, data.get('columns', df.select_dtypes(include=['number']).columns.tolist()))
        else:
            return jsonify({"error": "Invalid visualization type. Use 'bar', 'scatter', or 'heatmap'."}), 400
        
        return jsonify({"html": fig_html})
        
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/ai-insight', methods=['POST'])
def ai_insight():
    """Generate AI insights on the dataset using OpenAI."""
    try:
        data = request.json
        # Get dataset ID from the request data or session as fallback
        dataset_id = data.get('dataset_id') or session.get('current_dataset_id')
        
        if not dataset_id:
            return jsonify({"error": "No dataset found. Please upload a file first."}), 400
        
        # Get the dataset
        df = get_dataset(dataset_id)
        if df is None:
            return jsonify({"error": "Dataset not found"}), 404
        
        # Get dataset summary for OpenAI
        dataset_summary = wrangle.summarize_dataset(df)
        
        # Generate insights
        insights = ai.generate_insights(dataset_summary)
        
        return jsonify(insights)
        
    except Exception as e:
        logger.error(f"Error generating AI insights: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/detect-anomalies', methods=['POST'])
@require_login
def detect_anomalies():
    """Detect anomalies in the dataset using various methods."""
    try:
        data = request.json
        dataset_id = data.get('dataset_id') or session.get('current_dataset_id')
        method = data.get('method', 'zscore')
        threshold = data.get('threshold', 3.0)
        columns = data.get('columns')
        
        if not dataset_id:
            return jsonify({"error": "No dataset ID provided"}), 400
            
        df = get_dataset(dataset_id)
        if df is None:
            return jsonify({"error": "Dataset not found"}), 404
        
        # Detect anomalies
        anomalies = ai_enhance.detect_anomalies(df, method=method, threshold=threshold, columns=columns)
        
        return jsonify(anomalies)
    except Exception as e:
        logger.error(f"Error detecting anomalies: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/assess-data-quality', methods=['POST'])
@require_login
def assess_data_quality():
    """Assess data quality and provide cleaning recommendations."""
    try:
        data = request.json
        dataset_id = data.get('dataset_id') or session.get('current_dataset_id')
        
        if not dataset_id:
            return jsonify({"error": "No dataset ID provided"}), 400
            
        df = get_dataset(dataset_id)
        if df is None:
            return jsonify({"error": "Dataset not found"}), 404
        
        # Assess data quality
        quality_assessment = ai_enhance.assess_data_quality(df)
        
        return jsonify(quality_assessment)
    except Exception as e:
        logger.error(f"Error assessing data quality: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/automated-insights', methods=['POST'])
@require_login
def automated_insights():
    """Generate automated insights from the dataset."""
    try:
        data = request.json
        dataset_id = data.get('dataset_id') or session.get('current_dataset_id')
        sample_size = data.get('sample_size', 1000)
        
        if not dataset_id:
            return jsonify({"error": "No dataset ID provided"}), 400
            
        df = get_dataset(dataset_id)
        if df is None:
            return jsonify({"error": "Dataset not found"}), 404
        
        # Generate automated insights
        insights = ai_enhance.generate_automated_insights(df, sample_size=sample_size)
        
        return jsonify(insights)
    except Exception as e:
        logger.error(f"Error generating automated insights: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/analysis-recommendations', methods=['POST'])
@require_login
def analysis_recommendations():
    """Generate predictive analysis recommendations."""
    try:
        data = request.json
        dataset_id = data.get('dataset_id') or session.get('current_dataset_id')
        
        if not dataset_id:
            return jsonify({"error": "No dataset ID provided"}), 400
            
        df = get_dataset(dataset_id)
        if df is None:
            return jsonify({"error": "Dataset not found"}), 404
        
        # Generate analysis recommendations
        recommendations = ai_enhance.predict_analysis_recommendations(df)
        
        return jsonify(recommendations)
    except Exception as e:
        logger.error(f"Error generating analysis recommendations: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat_with_ai():
    """Chat with the AI about the dataset."""
    try:
        data = request.json
        # Get dataset ID from the request data or session as fallback
        dataset_id = data.get('dataset_id') or session.get('current_dataset_id')
        
        if not dataset_id or dataset_id not in DATASETS:
            return jsonify({"error": "No dataset found. Please upload a file first."}), 400
        
        # Get the dataset
        df = DATASETS[dataset_id]['current'].copy()
        
        # Get user message
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        # Get dataset summary for context
        dataset_summary = wrangle.summarize_dataset(df)
        
        # Generate response
        response = ai.chat_with_data(user_message, dataset_summary)
        
        return jsonify({"response": response})
        
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
@app.route('/nl-query', methods=['POST'])
def natural_language_query():
    """Execute a natural language query on the dataset."""
    try:
        data = request.json
        # Get dataset ID from the request data or session as fallback
        dataset_id = data.get('dataset_id') or session.get('current_dataset_id')
        
        if not dataset_id or dataset_id not in DATASETS:
            return jsonify({"error": "No dataset found. Please upload a file first."}), 400
        
        # Get the dataset
        df = DATASETS[dataset_id]['current'].copy()
        
        # Get user query
        query = data.get('query', '')
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        # Execute the natural language query
        result = ai.natural_language_query(query, df)
        
        if 'error' in result:
            return jsonify({"error": result['error']}), 400
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in natural language query: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
@app.route('/timeseries-forecast', methods=['POST'])
def timeseries_forecast():
    """Perform time series forecasting on the dataset."""
    try:
        data = request.json
        # Get dataset ID from the request data or session as fallback
        dataset_id = data.get('dataset_id') or session.get('current_dataset_id')
        
        if not dataset_id or dataset_id not in DATASETS:
            return jsonify({"error": "No dataset found. Please upload a file first."}), 400
        
        # Get the dataset
        df = DATASETS[dataset_id]['current'].copy()
        
        # Get parameters for the forecast
        date_column = data.get('date_column')
        target_column = data.get('target_column')
        forecast_periods = data.get('forecast_periods', 30)
        method = data.get('method', 'auto')
        
        if not date_column or not target_column:
            return jsonify({"error": "Date column and target column are required"}), 400
        
        # Validate parameters
        if date_column not in df.columns:
            return jsonify({"error": f"Date column '{date_column}' not found in the dataset"}), 400
            
        if target_column not in df.columns:
            return jsonify({"error": f"Target column '{target_column}' not found in the dataset"}), 400
            
        # Execute the time series forecast
        result = ml.time_series_forecast(df, date_column, target_column, forecast_periods, method)
        
        # Update the dataset with the forecast
        DATASETS[dataset_id]['forecast'] = result
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in time series forecasting: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
@app.route('/compare-datasets', methods=['POST'])
def compare_datasets():
    """Compare two datasets and provide detailed analysis."""
    try:
        data = request.json
        # Get the two dataset IDs to compare
        dataset_id1 = data.get('dataset_id1')
        dataset_id2 = data.get('dataset_id2')
        
        # Get the dataset names (optional)
        name1 = data.get('name1', "Dataset 1")
        name2 = data.get('name2', "Dataset 2")
        
        # Validate that both datasets exist
        if not dataset_id1 or dataset_id1 not in DATASETS:
            return jsonify({"error": f"Dataset 1 not found. Please upload a file first."}), 400
            
        if not dataset_id2 or dataset_id2 not in DATASETS:
            return jsonify({"error": f"Dataset 2 not found. Please upload a file first."}), 400
        
        # Get the datasets
        df1 = DATASETS[dataset_id1]['current'].copy()
        df2 = DATASETS[dataset_id2]['current'].copy()
        
        # Compare the datasets
        result = ml.compare_datasets(df1, df2, name1, name2)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error comparing datasets: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/data-preview', methods=['GET'])
def data_preview():
    """Get a preview of the current dataset."""
    try:
        # Get dataset_id from query parameter or session as fallback
        dataset_id = request.args.get('dataset_id') or session.get('current_dataset_id')
        
        if not dataset_id or dataset_id not in DATASETS:
            return jsonify({"error": "No dataset found. Please upload a file first."}), 400
        
        # Get the dataset
        df = DATASETS[dataset_id]['current']
        
        # Return first 10 rows as JSON
        preview = json.loads(df.head(10).to_json(orient='records'))
        
        return jsonify({
            "preview": preview,
            "columns": df.columns.tolist(),
            "total_rows": df.shape[0]
        })
        
    except Exception as e:
        logger.error(f"Error getting data preview: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/datasets', methods=['GET'])
def get_datasets():
    """Get datasets information."""
    try:
        # Get all datasets information
        datasets_info = {}
        for id, dataset in DATASETS.items():
            df = dataset['current']
            datasets_info[id] = {
                "name": dataset.get('filename', f"Dataset {id}"),
                "rows": len(df),
                "columns": df.columns.tolist()
            }
        
        # Get the current dataset for column information
        current_dataset_id = request.args.get('dataset_id') or session.get('current_dataset_id')
        if current_dataset_id and current_dataset_id in DATASETS:
            df = DATASETS[current_dataset_id]['current']
            return jsonify({
                "datasets": datasets_info,
                "columns": df.columns.tolist(),
                "dtypes": {col: str(df[col].dtype) for col in df.columns}
            })
        
        # Return all datasets if no current dataset
        return jsonify({
            "datasets": datasets_info
        })
    except Exception as e:
        logger.error(f"Error getting datasets: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/export', methods=['GET'])
def export_data():
    """Export the dataset in various formats."""
    try:
        # Get dataset_id from query parameter or session as fallback
        dataset_id = request.args.get('dataset_id') or session.get('current_dataset_id')
        format_type = request.args.get('format', 'csv')
        data_type = request.args.get('type', 'current')  # 'current', 'original', or 'clean'
        
        # If no dataset specified, use the first available one
        if not dataset_id and DATASETS:
            dataset_id = list(DATASETS.keys())[0]
            logger.info(f"No dataset ID provided, using first available: {dataset_id}")
        
        if not dataset_id or dataset_id not in DATASETS:
            return jsonify({"error": "No dataset found. Please upload a file first."}), 400
        
        # Get the dataset based on the requested type
        if data_type == 'original' and 'original' in DATASETS[dataset_id]:
            df = DATASETS[dataset_id]['original']
            filename = f"{DATASETS[dataset_id]['filename'].split('.')[0]}_original"
        elif data_type == 'clean':
            # Get current data and apply cleaning operations
            df = DATASETS[dataset_id]['current'].copy()
            # Handle the tuple return value from clean_dataset
            cleaned_result = wrangle.clean_dataset(df)
            # If it's a tuple (new version), extract the DataFrame
            if isinstance(cleaned_result, tuple) and len(cleaned_result) >= 1:
                df = cleaned_result[0]
            else:
                # For backward compatibility
                df = cleaned_result
            filename = f"{DATASETS[dataset_id]['filename'].split('.')[0]}_clean"
        else:
            # Default to current data
            df = DATASETS[dataset_id]['current']
            filename = DATASETS[dataset_id]['filename'].split('.')[0]  # Remove extension
        
        if format_type == 'csv':
            output = io.StringIO()
            df.to_csv(output, index=False)
            response = make_response(output.getvalue())
            response.headers["Content-Disposition"] = f"attachment; filename={filename}.csv"
            response.headers["Content-type"] = "text/csv"
            return response
        
        elif format_type == 'excel':
            output = io.BytesIO()
            df.to_excel(output, index=False, engine='openpyxl')
            output.seek(0)
            response = make_response(output.getvalue())
            response.headers["Content-Disposition"] = f"attachment; filename={filename}.xlsx"
            response.headers["Content-type"] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            return response
        
        elif format_type == 'json':
            response = make_response(df.to_json(orient='records'))
            response.headers["Content-Disposition"] = f"attachment; filename={filename}.json"
            response.headers["Content-type"] = "application/json"
            return response
        
        else:
            return jsonify({'error': 'Invalid export format'}), 400
            
    except Exception as e:
        logger.error(f"Error exporting data: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
@app.route('/export-insights', methods=['GET'])
def export_insights():
    """Export the AI insights in various formats."""
    try:
        # Get dataset_id from query parameter or session as fallback
        dataset_id = request.args.get('dataset_id') or session.get('current_dataset_id')
        format_type = request.args.get('format', 'text')
        
        # If no dataset specified, use the first available one
        if not dataset_id and DATASETS:
            dataset_id = list(DATASETS.keys())[0]
            logger.info(f"No dataset ID provided, using first available: {dataset_id}")
        
        if not dataset_id or dataset_id not in DATASETS:
            return jsonify({"error": "No dataset found. Please upload a file first."}), 400
        
        # Get the dataset for summary
        df = DATASETS[dataset_id]['current']
        filename = DATASETS[dataset_id]['filename'].split('.')[0]
        
        # Generate insights if not already available
        if 'insights' not in DATASETS[dataset_id]:
            # Create dataset summary for AI
            dataset_summary = wrangle.summarize_dataset(df)
            # Generate insights
            DATASETS[dataset_id]['insights'] = ai.generate_insights(dataset_summary)
            
        insights = DATASETS[dataset_id]['insights']
        insights_text = insights.get('insights_text', 'No insights available.')
        
        if format_type == 'text':
            response = Response(insights_text, mimetype='text/plain')
            response.headers["Content-Disposition"] = f"attachment; filename={filename}_insights.txt"
            return response
            
        elif format_type == 'html':
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>AI Insights for {filename}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
                    h1 {{ color: #333; }}
                    pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; white-space: pre-wrap; }}
                </style>
            </head>
            <body>
                <h1>AI Insights for {filename}</h1>
                <pre>{insights_text}</pre>
            </body>
            </html>
            """
            response = Response(html_content, mimetype='text/html')
            response.headers["Content-Disposition"] = f"attachment; filename={filename}_insights.html"
            return response
            
        elif format_type == 'pdf':
            # Return a JSON response with a message since we can't directly generate PDFs on the server
            # The client-side will handle PDF generation
            return jsonify({
                "message": "PDF generation is handled client-side. Please use the client's PDF export function.",
                "insights_text": insights_text,
                "filename": f"{filename}_insights.pdf"
            })
        
        else:
            return jsonify({"error": f"Unsupported format: {format_type}"}), 400
            
    except Exception as e:
        logger.error(f"Error exporting insights: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
@app.route('/export-visualization', methods=['GET'])
def export_visualization():
    """Export the visualization in various formats."""
    try:
        format_type = request.args.get('format', 'png')
        
        # This endpoint will mainly return instructions for client-side export
        # since Plotly visualizations are rendered client-side
        
        return jsonify({
            "message": "Visualization export is handled client-side. Please use the browser's save functionality.",
            "format": format_type
        })
        
    except Exception as e:
        logger.error(f"Error handling visualization export: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
@app.route('/export-forecast', methods=['GET'])
def export_forecast():
    """Export the time series forecast in various formats."""
    try:
        # Get dataset_id from query parameter or session as fallback
        dataset_id = request.args.get('dataset_id') or session.get('current_dataset_id')
        format_type = request.args.get('format', 'csv')
        
        # If no dataset specified, use the first available one
        if not dataset_id and DATASETS:
            dataset_id = list(DATASETS.keys())[0]
            logger.info(f"No dataset ID provided, using first available: {dataset_id}")
        
        if not dataset_id or dataset_id not in DATASETS or 'forecast' not in DATASETS[dataset_id]:
            return jsonify({"error": "No forecast found. Please run a forecast first."}), 400
        
        # Get the forecast data
        forecast_data = DATASETS[dataset_id]['forecast']
        filename = DATASETS[dataset_id]['filename'].split('.')[0]
        
        if format_type == 'csv':
            # Extract forecast DataFrame
            forecast_df = forecast_data.get('forecast_df')
            if forecast_df is None:
                return jsonify({"error": "Forecast data is not available"}), 400
                
            output = io.StringIO()
            forecast_df.to_csv(output, index=True)
            output.seek(0)
            
            response = Response(output.getvalue(), mimetype='text/csv')
            response.headers["Content-Disposition"] = f"attachment; filename={filename}_forecast.csv"
            return response
            
        elif format_type in ['chart', 'report']:
            # These formats are mainly handled client-side
            return jsonify({
                "message": f"{format_type.capitalize()} export is handled client-side. Please use the client's export function.",
                "format": format_type,
                "filename": f"{filename}_forecast"
            })
        
        else:
            return jsonify({"error": f"Unsupported format: {format_type}"}), 400
            
    except Exception as e:
        logger.error(f"Error exporting forecast: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
