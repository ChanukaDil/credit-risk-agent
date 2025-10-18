
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from enterprise_usage import EnterprisePreprocessor
from config_paths import DatasetConfig

def main():
    """Process your dataset"""
    
    print("="*70)
    print("PROCESSING YOUR BANK DATA")
    print("="*70)
    
    # Initialize
    processor = EnterprisePreprocessor()
    
    # Option 1: If dataset is in config
    # result = processor.process_kaggle_dataset('lending_club')
    
    # Option 2: If custom dataset (more common)
    from data_preprocessing import quick_process
    
    result = quick_process(
        csv_path='data/raw/Bank_data.csv',  # Corrected path to your CSV file
        target_col='RESCHEDULE',            # Change to your target column
        balance_strategy='smote_tomek',
        verify_leakage=True
    )
    
    if result:
        print("\n SUCCESS! Your data is processed!")
        print("\nNext steps:")
        print("1. Check data/processed/ for processed arrays")
        print("2. Check models/preprocessor/ for fitted preprocessor")
        print("3. Check results/reports/ for processing report")
        print("\n4. Now you can train your model!")

if __name__ == "__main__":
    main()