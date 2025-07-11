from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
import uuid
from datetime import datetime
import io
import tabula
import pdfplumber
from werkzeug.utils import secure_filename
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
import requests
import cloudinary
import cloudinary.uploader
import re
from pymongo import MongoClient

from llm_integration import LLMQueryProcessor
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)
CORS(app, resources={r"/api/*": {"origins": app.config['CORS_ORIGINS']}})

TEMP_DOWNLOAD_FOLDER = app.config['TEMP_DOWNLOAD_FOLDER']
ALLOWED_EXTENSIONS = app.config['ALLOWED_EXTENSIONS']
MAX_FILE_SIZE = app.config['MAX_CONTENT_LENGTH']

OPENROUTER_API_KEY = app.config['OPENROUTER_API_KEY']
OPENROUTER_MODEL = app.config['OPENROUTER_MODEL']
OPENROUTER_BASE_URL = app.config['OPENROUTER_BASE_URL']

CLOUDINARY_CLOUD_NAME = app.config['CLOUDINARY_CLOUD_NAME']
CLOUDINARY_API_KEY = app.config['CLOUDINARY_API_KEY']
CLOUDINARY_API_SECRET = app.config['CLOUDINARY_API_SECRET']

MONGO_URI = app.config['MONGO_URI']
MONGO_DB_NAME = app.config['MONGO_DB_NAME']

os.makedirs(TEMP_DOWNLOAD_FOLDER, exist_ok=True)

try:
    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET
    )
    logger.info("Cloudinary configured successfully.")
except Exception as e:
    logger.error(f"Failed to configure Cloudinary: {e}. Check CLOUDINARY_CLOUD_NAME, API_KEY, API_SECRET in .env")

mongo_client = None
db = None
try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client[MONGO_DB_NAME]
    mongo_client.admin.command('ismaster')
    logger.info(f"MongoDB Atlas connected successfully to database: {MONGO_DB_NAME}")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB Atlas: {e}. Check MONGO_URI in .env")
    mongo_client = None
    db = None

llm_processor = None
if OPENROUTER_API_KEY:
    try:
        llm_processor = LLMQueryProcessor(api_key=OPENROUTER_API_KEY, model=OPENROUTER_MODEL, base_url=OPENROUTER_BASE_URL)
        logger.info("LLMQueryProcessor (OpenRouter) initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize LLMQueryProcessor (OpenRouter): {e}. Queries will fallback to rule-based.")
else:
    logger.warning("OPENROUTER_API_KEY not found. LLM queries will use rule-based fallback.")

@dataclass
class DatasetInfo:
    id: str
    filename: str
    columns: List[str]
    row_count: int
    data_types: Dict[str, str]
    file_storage_url: str
    upload_time: datetime
    user_id: str

class FileProcessor:
    @staticmethod
    def allowed_file(filename: str) -> bool:
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    @staticmethod
    def process_csv(file_path: str) -> pd.DataFrame:
        try:
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    logger.info(f"Successfully read CSV with {encoding} encoding")
                    return df
                except UnicodeDecodeError:
                    continue
            raise ValueError("Unable to read CSV file with any supported encoding")
        except Exception as e:
            logger.error(f"Error processing CSV: {str(e)}")
            raise
    
    @staticmethod
    def process_excel(file_path: str) -> pd.DataFrame:
        try:
            excel_file = pd.ExcelFile(file_path)
            if len(excel_file.sheet_names) > 1:
                logger.info(f"Multiple sheets found: {excel_file.sheet_names}")
                sheet_name = excel_file.sheet_names[0]
            else:
                sheet_name = excel_file.sheet_names[0]
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            logger.info(f"Successfully read Excel sheet: {sheet_name}")
            return df
        except Exception as e:
            logger.error(f"Error processing Excel: {str(e)}")
            raise
    
    @staticmethod
    def process_pdf(file_path: str) -> pd.DataFrame:
        try:
            try:
                tables = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
                if tables and len(tables) > 0:
                    df = max(tables, key=len)
                    logger.info("Successfully extracted table using tabula")
                    return df
            except Exception as e:
                logger.warning(f"Tabula failed or not installed: {str(e)}")
            
            try:
                with pdfplumber.open(file_path) as pdf:
                    all_tables = []
                    for page in pdf.pages:
                        tables = page.extract_tables()
                        if tables:
                            cleaned_tables = []
                            for table in tables:
                                if table and any(row for row in table):
                                    cleaned_tables.append([[cell if cell is not None else '' for cell in row] for row in table])
                            if cleaned_tables:
                                all_tables.extend(cleaned_tables)
                    
                    if all_tables:
                        largest_table = max(all_tables, key=len)
                        if largest_table and largest_table[0] and all(len(row) == len(largest_table[0]) for row in largest_table[1:]):
                            df = pd.DataFrame(largest_table[1:], columns=largest_table[0])
                            logger.info("Successfully extracted table using pdfplumber")
                            return df
                        else:
                            raise ValueError("Extracted PDF table has invalid structure (e.g., empty header or inconsistent row lengths).")
            except Exception as e:
                logger.warning(f"PDFPlumber failed: {str(e)}")
            
            raise ValueError("Unable to extract tables from PDF")
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise

class DataAnalyzer:
    @staticmethod
    def analyze_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        try:
            df.columns = df.columns.astype(str).str.strip()
            data_types = {}
            for col in df.columns:
                dtype = str(df[col].dtype)
                if dtype.startswith('int') or dtype.startswith('float'):
                    data_types[col] = 'numeric'
                elif dtype == 'object':
                    try:
                        pd.to_datetime(df[col].dropna().iloc[:5])
                        data_types[col] = 'date'
                    except (ValueError, TypeError):
                        data_types[col] = 'text'
                else:
                    data_types[col] = 'text'
            
            sample_data = df.head(5).fillna('').to_dict('records')
            
            return {
                'columns': df.columns.tolist(),
                'row_count': len(df),
                'data_types': data_types,
                'sample_data': sample_data,
                'summary_stats': DataAnalyzer._get_summary_stats(df)
            }
        except Exception as e:
            logger.error(f"Error analyzing dataframe: {str(e)}")
            raise
    
    @staticmethod
    def _get_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats = {}
        for col in numeric_cols:
            stats[col] = {
                'mean': float(df[col].mean()) if not df[col].isna().all() else 0,
                'sum': float(df[col].sum()) if not df[col].isna().all() else 0,
                'min': float(df[col].min()) if not df[col].isna().all() else 0,
                'max': float(df[col].max()) if not df[col].isna().all() else 0,
                'count': int(df[col].count())
            }
        return stats
    
    @staticmethod
    def process_nlp_query(query: str, df: pd.DataFrame, dataset_info: Dict) -> Dict[str, Any]:
        logger.info("Using rule-based NLP query processing (fallback).")
        query_lower = query.lower()
        
        if 'top' in query_lower and ('product' in query_lower or 'item' in query_lower):
            product_col = next((c for c, d in dataset_info['data_types'].items() if d == 'text'), None)
            sales_col = next((c for c, d in dataset_info['data_types'].items() if d == 'numeric'), None)
            
            if product_col and sales_col:
                product_sales = df.groupby(product_col)[sales_col].sum().sort_values(ascending=False).head(10)
                data = [{'name': str(k), 'value': float(v)} for k, v in product_sales.items()]
                return {
                    'query_type': 'comparison',
                    'chart_type': 'bar',
                    'data': data,
                    'title': f'Top {product_col.title()} by {sales_col.title()}',
                    'insight': f"Top performing {product_col} is {data[0]['name']} with {data[0]['value']:,.0f}" if data else "No data",
                    'x_axis': 'name',
                    'y_axis': 'value'
                }
        elif 'region' in query_lower or 'location' in query_lower:
            region_col = next((c for c in dataset_info['columns'] if 'region' in c.lower() or 'location' in c.lower()), None)
            sales_col = next((c for c, d in dataset_info['data_types'].items() if d == 'numeric'), None)

            if region_col and sales_col:
                region_sales = df.groupby(region_col)[sales_col].sum().sort_values(ascending=False)
                data = [{'name': str(k), 'value': float(v)} for k, v in region_sales.items()]
                return {
                    'query_type': 'comparison',
                    'chart_type': 'pie',
                    'data': data,
                    'title': f'{sales_col.title()} Distribution by {region_col.title()}',
                    'insight': f"Leading {region_col} is {data[0]['name']} with {data[0]['value']:,.0f}" if data else "No data",
                    'x_axis': 'name',
                    'y_axis': 'value'
                }
        elif 'trend' in query_lower or 'monthly' in query_lower or 'time' in query_lower:
            date_col = next((c for c, d in dataset_info['data_types'].items() if d == 'date'), None)
            sales_col = next((c for c, d in dataset_info['data_types'].items() if d == 'numeric'), None)
            
            if date_col and sales_col:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df_filtered = df.dropna(subset=[date_col])
                
                monthly_sales = df_filtered.groupby(df_filtered[date_col].dt.to_period('M'))[sales_col].sum().sort_index()
                data = [{'name': str(k), 'value': float(v)} for k, v in monthly_sales.items()]
                return {
                    'query_type': 'trend',
                    'chart_type': 'line',
                    'data': data,
                    'title': f'{sales_col.title()} Trend Over Time',
                    'insight': 'Sales trend analysis showing changes over time.' if data else "No data",
                    'x_axis': 'name',
                    'y_axis': 'value'
                }
        
        total_sales = df.select_dtypes(include=[np.number]).sum().sum()
        total_quantity = df.select_dtypes(include=[np.number]).sum().sum()
        
        return {
            'query_type': 'summary',
            'chart_type': 'table',
            'data': [{'Metric': 'Total Sales', 'Value': total_sales.toLocaleString() if isinstance(total_sales, (int, float)) else total_sales}],
            'title': 'Data Summary',
            'insight': f'Your dataset contains {len(df)} records with total numeric sum of ${total_sales.toLocaleString() if isinstance(total_sales, (int, float)) else total_sales}.'
        }

@app.route('/api/analyze_file_metadata', methods=['POST'])
def analyze_file_metadata():
    try:
        if db is None:
            return jsonify({'error': 'MongoDB not connected.'}), 500

        data = request.json
        file_storage_url = data.get('file_storage_url')
        filename = data.get('filename')
        file_extension = data.get('file_extension')
        dataset_id = data.get('dataset_id')
        user_id = data.get('userId', 'anonymous')

        if not file_storage_url or not filename or not file_extension or not dataset_id:
            return jsonify({'error': 'Missing required parameters for file analysis.'}), 400

        local_filename = f"{dataset_id}_{secure_filename(filename)}"
        local_file_path = os.path.join(TEMP_DOWNLOAD_FOLDER, local_filename)

        logger.info(f"Attempting to download file from: {file_storage_url}")
        response = requests.get(file_storage_url, stream=True)
        response.raise_for_status()

        with open(local_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Downloaded file to temporary path: {local_file_path}")

        df = None
        if file_extension == 'csv':
            df = FileProcessor.process_csv(local_file_path)
        elif file_extension in ['xlsx', 'xls']:
            df = FileProcessor.process_excel(local_file_path)
        elif file_extension == 'pdf':
            df = FileProcessor.process_pdf(local_file_path)
        else:
            return jsonify({'error': 'Unsupported file type for analysis.'}), 400
        
        try:
            os.remove(local_file_path)
            logger.info(f"Cleaned up temporary file: {local_file_path}")
        except Exception as cleanup_e:
            logger.warning(f"Failed to clean up temporary file {local_file_path}: {cleanup_e}")

        if df is None:
            return jsonify({'error': 'Failed to process file for analysis.'}), 500

        analysis = DataAnalyzer.analyze_dataframe(df)

        dataset_metadata = {
            "id": dataset_id,
            "filename": filename,
            "file_storage_url": file_storage_url,
            "columns": json.dumps(analysis['columns']),
            "row_count": analysis['row_count'],
            "data_types": json.dumps(analysis['data_types']),
            "upload_time": datetime.utcnow(),
            "user_id": user_id
        }

        db.datasets.insert_one(dataset_metadata)
        logger.info(f"Dataset metadata saved to MongoDB: {dataset_id}")

        return jsonify({
            'success': True,
            'dataset_id': dataset_id,
            'filename': filename,
            'columns': analysis['columns'],
            'row_count': analysis['row_count'],
            'data_types': analysis['data_types'],
            'sample_data': analysis['sample_data'][:3],
            'summary_stats': analysis['summary_stats']
        })

    except requests.exceptions.RequestException as req_err:
        logger.error(f"Network/Download error during file analysis: {req_err}")
        return jsonify({'error': f'Failed to download file from Cloudinary: {req_err}'}), 500
    except Exception as e:
        logger.error(f"File metadata analysis error: {str(e)}", exc_info=True) # Log traceback
        return jsonify({'error': f'File analysis failed: {str(e)}'}), 500

@app.route('/api/query', methods=['POST'])
def process_query():
    try:
        if db is None:
            return jsonify({'error': 'MongoDB not connected.'}), 500

        data = request.json
        query = data.get('query', '')
        dataset_id = data.get('dataset_id', '')
        user_id = data.get('userId', 'anonymous')
        
        if not query or not dataset_id:
            return jsonify({'error': 'Query and dataset_id are required'}), 400
        
        dataset_info_doc = db.datasets.find_one({"id": dataset_id, "user_id": user_id})
        if not dataset_info_doc:
            return jsonify({'error': 'Dataset not found or access denied.'}), 404
        
        dataset_info = {
            "id": dataset_info_doc["id"],
            "filename": dataset_info_doc["filename"],
            "file_storage_url": dataset_info_doc["file_storage_url"],
            "columns": json.loads(dataset_info_doc["columns"]),
            "row_count": dataset_info_doc["row_count"],
            "data_types": json.loads(dataset_info_doc["data_types"]),
            "upload_time": dataset_info_doc["upload_time"],
            "user_id": dataset_info_doc["user_id"]
        }

        file_storage_url = dataset_info['file_storage_url']
        local_filename = f"{dataset_id}_{secure_filename(dataset_info['filename'])}"
        local_file_path = os.path.join(TEMP_DOWNLOAD_FOLDER, local_filename)

        logger.info(f"Attempting to download file from: {file_storage_url}")
        response = requests.get(file_storage_url, stream=True)
        response.raise_for_status()

        with open(local_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Downloaded file to temporary path: {local_file_path}")

        file_ext = dataset_info['filename'].rsplit('.', 1)[1].lower()
        
        df = None
        if file_ext == 'csv':
            df = FileProcessor.process_csv(local_file_path)
        elif file_ext in ['xlsx', 'xls']:
            df = FileProcessor.process_excel(local_file_path)
        elif file_ext == 'pdf':
            df = FileProcessor.process_pdf(local_file_path)
        else:
            return jsonify({'error': 'Unsupported file type for query processing'}), 400
        
        try:
            os.remove(local_file_path)
            logger.info(f"Cleaned up temporary file: {local_file_path}")
        except Exception as cleanup_e:
            logger.warning(f"Failed to clean up temporary file {local_file_path}: {cleanup_e}")

        if df is None:
            return jsonify({'error': 'Failed to load data for query processing'}), 500

        if llm_processor:
            result = llm_processor.process_query(query, df, dataset_info)
        else:
            result = DataAnalyzer.process_nlp_query(query, df, dataset_info)
            if hasattr(result, '__dict__'): # Ensure it's a dict for JSON serialization
                result = result.__dict__
            
        return jsonify({
            'success': True,
            'result': result
        })
        
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Network/Download error during query processing: {req_err}")
        return jsonify({'error': f'Failed to download file from Cloudinary: {req_err}'}), 500
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}", exc_info=True) # Log traceback
        return jsonify({'error': f'Query processing failed: {str(e)}'}), 500

@app.route('/api/datasets', methods=['POST'])
def get_datasets():
    try:
        if db is None:
            return jsonify({'error': 'MongoDB not connected.'}), 500

        data = request.json
        user_id = data.get('userId', 'anonymous')

        datasets_cursor = db.datasets.find({"user_id": user_id}).sort("upload_time", -1).limit(10)
        
        datasets = []
        for doc in datasets_cursor:
            doc_id = str(doc.get('_id'))
            datasets.append({
                'id': doc.get('id'),
                'filename': doc.get('filename'),
                'row_count': doc.get('row_count'),
                'upload_time': doc.get('upload_time').isoformat() if doc.get('upload_time') else None
            })
        
        return jsonify({
            'success': True,
            'datasets': datasets
        })
        
    except Exception as e:
        logger.error(f"Error getting datasets from MongoDB: {str(e)}", exc_info=True) # Log traceback
        return jsonify({'error': f'Failed to get datasets: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

if __name__ == '__main__':
        # Use Gunicorn in production, Flask's built-in server for local development
        # Heroku sets the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=app.config['DEBUG'], host='0.0.0.0', port=port)
    

