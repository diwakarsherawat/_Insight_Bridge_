import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration class"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # File processing settings (temp folder for backend to download files)
    TEMP_DOWNLOAD_FOLDER = os.environ.get('TEMP_DOWNLOAD_FOLDER', 'temp_downloads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'pdf'}
    
    # Cloudinary settings
    CLOUDINARY_CLOUD_NAME = os.environ.get('CLOUDINARY_CLOUD_NAME')
    CLOUDINARY_API_KEY = os.environ.get('CLOUDINARY_API_KEY')
    CLOUDINARY_API_SECRET = os.environ.get('CLOUDINARY_API_SECRET')
    
    # MongoDB Atlas settings
    MONGO_URI = os.environ.get('MONGO_URI')
    MONGO_DB_NAME = os.environ.get('MONGO_DB_NAME', 'nlp_bi_db') # Default DB name

    # OpenRouter settings (NEW)
    OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY') # Your OpenRouter API key
    OPENROUTER_BASE_URL = os.environ.get('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1/chat/completions') # OpenRouter API endpoint
    OPENROUTER_MODEL = os.environ.get('OPENROUTER_MODEL', 'openai/gpt-3.5-turbo') # Default model, e.g., 'openai/gpt-3.5-turbo' or 'mistralai/mistral-7b-instruct'
    
    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'app.log')
    
    
        # In production, replace '*' with your actual Netlify frontend URL (e.g., 'https://your-app-name.netlify.app')
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
    # Processing settings
    MAX_PROCESSING_TIME = int(os.environ.get('MAX_PROCESSING_TIME', 300))  # 5 minutes
    CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 10000))  # For large file processing
    
    # NLP settings
    USE_LOCAL_MODEL = os.environ.get('USE_LOCAL_MODEL', 'False').lower() == 'true'
    LOCAL_MODEL_PATH = os.environ.get('LOCAL_MODEL_PATH', 'models/local_nlp_model')
    
    # Export settings
    EXPORT_FOLDER = os.environ.get('EXPORT_FOLDER', 'exports')
    EXPORT_FORMATS = ['pdf', 'excel', 'csv', 'json']
    
    @staticmethod
    def init_app(app):
        """Initialize the Flask app with configuration"""
        # Create necessary directories
        os.makedirs(Config.TEMP_DOWNLOAD_FOLDER, exist_ok=True) # Temp folder for backend downloads
        os.makedirs(Config.EXPORT_FOLDER, exist_ok=True)
        
        # Set up logging
        import logging
        from logging.handlers import RotatingFileHandler
        
        if not app.debug:
            file_handler = RotatingFileHandler(
                Config.LOG_FILE, 
                maxBytes=10240000,  # 10MB
                backupCount=10
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
            ))
            file_handler.setLevel(getattr(logging, Config.LOG_LEVEL))
            app.logger.addHandler(file_handler)
            app.logger.setLevel(getattr(logging, Config.LOG_LEVEL))
            app.logger.info('NLP BI Backend startup')

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Enhanced security for production
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    TEMP_DOWNLOAD_FOLDER = 'test_temp_downloads'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
