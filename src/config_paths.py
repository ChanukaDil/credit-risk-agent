
import os
from pathlib import Path

# Base project directory
PROJECT_ROOT = Path(__file__).parent.parent

class ProjectPaths:
    """Centralized path management for the project"""
    
    # =====================================================
    # DATA PATHS
    # =====================================================
    
    # Raw data (input files - CSV, JSON, etc.)
    DATA_RAW = PROJECT_ROOT / "data" / "raw"
    
    # Processed data (cleaned, engineered features)
    DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
    
    # Temporary data (intermediate processing)
    DATA_TEMP = PROJECT_ROOT / "data" / "temp"
    
    # =====================================================
    # MODEL PATHS
    # =====================================================
    
    # Preprocessors (scalers, encoders, imputers)
    MODELS_PREPROCESSOR = PROJECT_ROOT / "models" / "preprocessor"
    
    # Autoencoder models
    MODELS_AUTOENCODER = PROJECT_ROOT / "models" / "autoencoder"
    
    # Vector database models
    MODELS_VECTOR_DB = PROJECT_ROOT / "models" / "vector_db"
    
    # Model checkpoints
    MODELS_CHECKPOINTS = PROJECT_ROOT / "models" / "checkpoints"
    
    # =====================================================
    # RESULTS PATHS
    # =====================================================
    
    # Main results directory
    RESULTS = PROJECT_ROOT / "results"
    
    # Visualizations (plots, charts)
    RESULTS_VISUALIZATIONS = PROJECT_ROOT / "results" / "visualizations"
    
    # Metrics and evaluation
    RESULTS_METRICS = PROJECT_ROOT / "results" / "metrics"
    
    # Reports (HTML, PDF)
    RESULTS_REPORTS = PROJECT_ROOT / "results" / "reports"
    
    # Logs
    RESULTS_LOGS = PROJECT_ROOT / "results" / "logs"
    
    # =====================================================
    # CONFIG PATHS
    # =====================================================
    
    # Configuration files
    CONFIG = PROJECT_ROOT / "config"
    
    # Notebooks
    NOTEBOOKS = PROJECT_ROOT / "notebooks"
    
    # Tests
    TESTS = PROJECT_ROOT / "tests"
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.DATA_RAW,
            cls.DATA_PROCESSED,
            cls.DATA_TEMP,
            cls.MODELS_PREPROCESSOR,
            cls.MODELS_AUTOENCODER,
            cls.MODELS_VECTOR_DB,
            cls.MODELS_CHECKPOINTS,
            cls.RESULTS,
            cls.RESULTS_VISUALIZATIONS,
            cls.RESULTS_METRICS,
            cls.RESULTS_REPORTS,
            cls.RESULTS_LOGS,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(" All project directories created successfully!")
    
    @classmethod
    def get_data_path(cls, dataset_name: str, data_type: str = "raw") -> Path:
        """
        Get path for specific dataset
        
        Args:
            dataset_name: Name of dataset (e.g., 'kaggle_credit', 'banking_data')
            data_type: Type of data ('raw' or 'processed')
            
        Returns:
            Path object
        """
        if data_type == "raw":
            return cls.DATA_RAW / f"{dataset_name}.csv"
        elif data_type == "processed":
            return cls.DATA_PROCESSED / dataset_name
        else:
            raise ValueError("data_type must be 'raw' or 'processed'")
    
    @classmethod
    def get_model_path(cls, model_name: str, model_type: str = "preprocessor") -> Path:
        """
        Get path for specific model
        
        Args:
            model_name: Name of model
            model_type: Type of model ('preprocessor', 'autoencoder', 'vector_db')
            
        Returns:
            Path object
        """
        if model_type == "preprocessor":
            return cls.MODELS_PREPROCESSOR / f"{model_name}_preprocessor.pkl"
        elif model_type == "autoencoder":
            return cls.MODELS_AUTOENCODER / f"{model_name}_autoencoder.pkl"
        elif model_type == "vector_db":
            return cls.MODELS_VECTOR_DB / f"{model_name}_vector_db.pkl"
        else:
            raise ValueError("model_type must be 'preprocessor', 'autoencoder', or 'vector_db'")
    
    @classmethod
    def get_results_path(cls, file_name: str, result_type: str = "visualizations") -> Path:
        """
        Get path for results file
        
        Args:
            file_name: Name of file
            result_type: Type of result ('visualizations', 'metrics', 'reports', 'logs')
            
        Returns:
            Path object
        """
        if result_type == "visualizations":
            return cls.RESULTS_VISUALIZATIONS / file_name
        elif result_type == "metrics":
            return cls.RESULTS_METRICS / file_name
        elif result_type == "reports":
            return cls.RESULTS_REPORTS / file_name
        elif result_type == "logs":
            return cls.RESULTS_LOGS / file_name
        else:
            raise ValueError("result_type must be 'visualizations', 'metrics', 'reports', or 'logs'")


# =====================================================
# DATASET CONFIGURATIONS
# =====================================================

class DatasetConfig:
    """Configuration for different datasets"""
    
    # Common Kaggle Credit Risk datasets
    KAGGLE_DATASETS = {
        "lending_club": {
            "filename": "lending_club_loan_data.csv",
            "target_column": "loan_status",
            "description": "Lending Club Loan Data"
        },
        "credit_risk": {
            "filename": "credit_risk_dataset.csv", 
            "target_column": "default",
            "description": "Credit Risk Dataset"
        },
        "home_credit": {
            "filename": "home_credit_default_risk.csv",
            "target_column": "target",
            "description": "Home Credit Default Risk"
        }
    }
    
    # Banking institute datasets
    BANKING_DATASETS = {
        "bank_marketing": {
            "filename": "bank_marketing.csv",
            "target_column": "y",
            "description": "Bank Marketing Campaign"
        },
        "german_credit": {
            "filename": "german_credit_data.csv",
            "target_column": "default",
            "description": "German Credit Data"
        }
    }
    
    @classmethod
    def get_dataset_info(cls, dataset_name: str) -> dict:
        """Get dataset configuration"""
        all_datasets = {**cls.KAGGLE_DATASETS, **cls.BANKING_DATASETS}
        
        if dataset_name in all_datasets:
            return all_datasets[dataset_name]
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")


# =====================================================
# USAGE EXAMPLES
# =====================================================

if __name__ == "__main__":
    
    print(" ENTERPRISE PATH CONFIGURATION")
    print("=" * 50)
    
    # Create all directories
    ProjectPaths.create_directories()
    
    # Example path usage
    print("\n EXAMPLE PATHS:")
    print(f"Raw Data: {ProjectPaths.DATA_RAW}")
    print(f"Processed Data: {ProjectPaths.DATA_PROCESSED}")
    print(f"Preprocessor Models: {ProjectPaths.MODELS_PREPROCESSOR}")
    print(f"Results: {ProjectPaths.RESULTS}")
    print(f"Visualizations: {ProjectPaths.RESULTS_VISUALIZATIONS}")
    
    # Dataset-specific paths
    print("\n DATASET PATHS:")
    kaggle_path = ProjectPaths.get_data_path("kaggle_credit_data", "raw")
    print(f"Kaggle Data: {kaggle_path}")
    
    preprocessor_path = ProjectPaths.get_model_path("kaggle_credit", "preprocessor")
    print(f"Preprocessor: {preprocessor_path}")
    
    viz_path = ProjectPaths.get_results_path("imbalance_comparison.png", "visualizations")
    print(f"Visualization: {viz_path}")
    
    # Dataset configurations
    print("\n AVAILABLE DATASETS:")
    for name, config in DatasetConfig.KAGGLE_DATASETS.items():
        print(f"  {name}: {config['description']} (target: {config['target_column']})")