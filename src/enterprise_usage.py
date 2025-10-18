
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from config_paths import ProjectPaths, DatasetConfig
from data_preprocessing import CreditRiskPreprocessor, quick_process, process_multiple_datasets

class EnterprisePreprocessor:
    """
    Enterprise wrapper for the preprocessing pipeline with proper path management
    """
    
    def __init__(self):
        # Create all necessary directories
        ProjectPaths.create_directories()
        self.paths = ProjectPaths()
        
    def process_kaggle_dataset(self, dataset_name: str) -> dict:
        """
        Process a Kaggle dataset with enterprise folder structure
        
        Args:
            dataset_name: Name from DatasetConfig.KAGGLE_DATASETS
            
        Returns:
            Processing results dictionary
        """
        print(f"\n{'='*70}")
        print(f" PROCESSING KAGGLE DATASET: {dataset_name.upper()}")
        print(f"{'='*70}")
        
        # Get dataset configuration
        try:
            config = DatasetConfig.get_dataset_info(dataset_name)
        except ValueError as e:
            print(f"❌ Error: {e}")
            return None
        
        # Define paths using enterprise structure
        raw_data_path = self.paths.get_data_path(config['filename'].replace('.csv', ''), 'raw')
        preprocessor_path = self.paths.get_model_path(dataset_name, 'preprocessor')
        
        print(f" Dataset: {config['description']}")
        print(f" Raw Data Path: {raw_data_path}")
        print(f" Preprocessor Path: {preprocessor_path}")
        print(f" Target Column: {config['target_column']}")
        
        # Check if file exists
        if not raw_data_path.exists():
            print(f" File not found: {raw_data_path}")
            print(f" Please place your {config['filename']} in {self.paths.DATA_RAW}")
            return None
        
        # Process with enterprise paths
        try:
            splits, preprocessor = quick_process(
                csv_path=str(raw_data_path),
                target_col=config['target_column'],
                balance_strategy='smote_tomek',
                save_path=str(preprocessor_path),
                verify_leakage=True
            )
            
            # Save processed data with proper naming
            self._save_processed_data(splits, dataset_name)
            
            # Generate enterprise report
            self._generate_processing_report(splits, dataset_name, config)
            
            return {
                'splits': splits,
                'preprocessor': preprocessor,
                'config': config,
                'paths': {
                    'raw_data': raw_data_path,
                    'preprocessor': preprocessor_path,
                    'processed_data': self.paths.DATA_PROCESSED / dataset_name
                }
            }
            
        except Exception as e:
            print(f" Processing failed: {e}")
            return None
    
    def process_banking_dataset(self, dataset_name: str) -> dict:
        """
        Process a banking dataset with enterprise folder structure
        
        Args:
            dataset_name: Name from DatasetConfig.BANKING_DATASETS
            
        Returns:
            Processing results dictionary
        """
        print(f"\n{'='*70}")
        print(f" PROCESSING BANKING DATASET: {dataset_name.upper()}")
        print(f"{'='*70}")
        
        # Similar to Kaggle processing but for banking data
        return self.process_kaggle_dataset(dataset_name)  # Same logic applies
    
    def process_multiple_datasets_enterprise(self, dataset_configs: dict) -> dict:
        """
        Process multiple datasets with enterprise structure
        
        Args:
            dataset_configs: Dictionary with dataset names and types
            Example:
            {
                'kaggle_lending': {'type': 'kaggle', 'name': 'lending_club'},
                'banking_german': {'type': 'banking', 'name': 'german_credit'}
            }
            
        Returns:
            Results for all datasets
        """
        print(f"\n{'#'*70}")
        print(" ENTERPRISE MULTI-DATASET PROCESSING")
        print(f"{'#'*70}")
        
        results = {}
        
        for alias, config in dataset_configs.items():
            print(f"\n Processing: {alias}")
            
            if config['type'] == 'kaggle':
                result = self.process_kaggle_dataset(config['name'])
            elif config['type'] == 'banking':
                result = self.process_banking_dataset(config['name'])
            else:
                print(f" Unknown dataset type: {config['type']}")
                continue
            
            if result:
                results[alias] = result
                print(f" {alias} processed successfully!")
            else:
                print(f" {alias} processing failed!")
        
        # Generate comparative report
        if results:
            self._generate_comparative_report(results)
        
        return results
    
    def _save_processed_data(self, splits: dict, dataset_name: str):
        """Save processed data with proper enterprise naming"""
        processed_dir = self.paths.DATA_PROCESSED / dataset_name
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Save numpy arrays
        for split_name, data in splits.items():
            save_path = processed_dir / f"{split_name}.npy"
            np.save(save_path, data)
        
        # Save metadata
        metadata = {
            'dataset_name': dataset_name,
            'splits': list(splits.keys()),
            'shapes': {k: v.shape for k, v in splits.items()},
            'processed_date': pd.Timestamp.now().isoformat()
        }
        
        metadata_path = processed_dir / "metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f" Processed data saved to: {processed_dir}")
    
    def _generate_processing_report(self, splits: dict, dataset_name: str, config: dict):
        """Generate processing report"""
        from collections import Counter
        
        report_path = self.paths.get_results_path(f"{dataset_name}_processing_report.txt", "reports")
        
        with open(report_path, 'w') as f:
            f.write(f"CREDIT RISK PREPROCESSING REPORT\n")
            f.write(f"={'='*50}\n\n")
            f.write(f"Dataset: {config['description']}\n")
            f.write(f"Target Column: {config['target_column']}\n")
            f.write(f"Processing Date: {pd.Timestamp.now()}\n\n")
            
            f.write(f"DATA SPLITS:\n")
            f.write(f"{'-'*20}\n")
            for split_name, data in splits.items():
                if split_name.startswith('X_'):
                    continue
                y_data = data
                f.write(f"{split_name}: {len(y_data):,} samples - {dict(Counter(y_data))}\n")
            
            f.write(f"\nFEATURE SHAPES:\n")
            f.write(f"{'-'*20}\n")
            for split_name, data in splits.items():
                if split_name.startswith('X_'):
                    f.write(f"{split_name}: {data.shape}\n")
        
        print(f" Processing report saved to: {report_path}")
    
    def _generate_comparative_report(self, results: dict):
        """Generate comparative report across datasets"""
        report_path = self.paths.get_results_path("comparative_analysis.txt", "reports")
        
        with open(report_path, 'w') as f:
            f.write(f"MULTI-DATASET COMPARATIVE ANALYSIS\n")
            f.write(f"={'='*60}\n\n")
            
            for alias, result in results.items():
                if result:
                    splits = result['splits']
                    config = result['config']
                    
                    f.write(f"Dataset: {alias.upper()}\n")
                    f.write(f"Description: {config['description']}\n")
                    f.write(f"Target: {config['target_column']}\n")
                    f.write(f"Training samples: {len(splits['y_train']):,}\n")
                    f.write(f"Test samples: {len(splits['y_test']):,}\n")
                    f.write(f"Features: {splits['X_train'].shape[1]}\n")
                    f.write(f"{'-'*40}\n\n")
        
        print(f" Comparative report saved to: {report_path}")


# =====================================================
# USAGE EXAMPLES
# =====================================================

def example_single_dataset():
    """Example: Process single Kaggle dataset"""
    print(" EXAMPLE 1: Single Dataset Processing")
    
    processor = EnterprisePreprocessor()
    
    # Process a Kaggle dataset (you need to place the CSV file in data/raw/)
    result = processor.process_kaggle_dataset('lending_club')
    
    if result:
        splits = result['splits']
        print(f"\n SUCCESS! Data ready for training:")
        print(f"   Training: {splits['X_train'].shape}")
        print(f"   Testing: {splits['X_test'].shape}")


def example_multiple_datasets():
    """Example: Process multiple datasets"""
    print(" EXAMPLE 2: Multiple Dataset Processing")
    
    processor = EnterprisePreprocessor()
    
    # Define datasets to process
    datasets = {
        'kaggle_lending': {'type': 'kaggle', 'name': 'lending_club'},
        'kaggle_credit': {'type': 'kaggle', 'name': 'credit_risk'},
        'banking_german': {'type': 'banking', 'name': 'german_credit'}
    }
    
    # Process all datasets
    results = processor.process_multiple_datasets_enterprise(datasets)
    
    print(f"\n Processed {len(results)} datasets successfully!")


def example_custom_dataset():
    """Example: Process custom dataset with manual configuration"""
    print(" EXAMPLE 3: Custom Dataset Processing")
    
    # For datasets not in the predefined configurations
    processor = EnterprisePreprocessor()
    
    # Place your custom CSV in data/raw/
    custom_path = processor.paths.DATA_RAW / "Bank_data.xlsx"
    
    if custom_path.exists():
        splits, preprocessor = quick_process(
            csv_path=str(custom_path),
            target_col=None,  # Auto-detect
            balance_strategy='smote_tomek',
            save_path=str(processor.paths.get_model_path('custom', 'preprocessor'))
        )
        
        print(" Custom dataset processed!")
    else:
        print(f" Place your CSV file at: {custom_path}")


def main():
    """Main execution with enterprise setup"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║           🏢 ENTERPRISE CREDIT RISK PREPROCESSING               ║
    ║               with Professional Folder Structure                 ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize enterprise processor
    processor = EnterprisePreprocessor()
    
    print("\n FOLDER STRUCTURE CREATED:")
    print(f"   Raw Data: {processor.paths.DATA_RAW}")
    print(f"   Processed: {processor.paths.DATA_PROCESSED}")
    print(f"   Models: {processor.paths.MODELS_PREPROCESSOR}")
    print(f"   Results: {processor.paths.RESULTS}")
    
    print("\n AVAILABLE DATASET CONFIGURATIONS:")
    for name, config in DatasetConfig.KAGGLE_DATASETS.items():
        print(f"   {name}: {config['description']}")
    
    print("\n TO GET STARTED:")
    print("   1. Place your CSV files in data/raw/")
    print("   2. Run: python enterprise_usage.py")
    print("   3. Check results in data/processed/ and results/")
    
    print("\n READY FOR ENTERPRISE-LEVEL PREPROCESSING!")


if __name__ == "__main__":
    main()
    
    # Uncomment to run examples:
    # example_single_dataset()
    # example_multiple_datasets()
    # example_custom_dataset()