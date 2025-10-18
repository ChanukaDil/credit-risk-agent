#!/usr/bin/env python3
"""
Test the complete credit risk system with actual Bank data features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')

from risk_scoring import CreditRiskScorer

def test_system():
    """Test the complete system with realistic Bank data"""
    
    print("="*70)
    print("🧪 TESTING COMPLETE CREDIT RISK SYSTEM")
    print("="*70)
    
    # Step 1: Initialize scorer
    print("\n📂 Loading trained models...")
    try:
        scorer = CreditRiskScorer(
            model_path='models/autoencoder/default_autoencoder.pth',
            preprocessor_path='models/preprocessor/preprocessor.pkl'
        )
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return
    
    # Step 2: Calibrate error range
    print("\n⚖️ Calibrating risk scoring...")
    try:
        val_errors = np.load('results/validation_errors.npy')
        scorer.calibrate_error_range(val_errors)
        print("✅ Risk scoring calibrated!")
    except Exception as e:
        print(f"❌ Failed to calibrate: {e}")
        return
    
    # Step 3: Create test customer with actual Bank data structure
    print("\n👤 Creating test customer application...")
    
    # Create a customer using the actual feature names from Bank_data
    customer = pd.DataFrame({
        # Numerical features (these will be kept as-is)
        'NET_RENTAL': [12000],
        'NO_OF_RENTAL': [36],
        'PAID_RENTALS': [24],
        'CB_ARREARS_AGE': [0],
        'YOM': [2010],
        'FINANCE_AMOUNT': [300000],
        'CUSTOMER_VALUATION': [400000],
        'EFFECTIVE_RATE': [25.5],
        'AGE': [35],
        'INCOME': [50000],
        'EXPENSE': [15000],
        
        # Categorical features (these will be encoded)
        'PRODUCT_CODE': ['HP'],
        'PRODUCT_NAME': ['HIRE PURCHASE'],
        'PRODUCT_CATEGORY': ['HIRE PURCHASE'],
        'CONTRACT_NO': ['TEST001'],
        'CONTRACT_STATUS': ['S'],
        'CONTRACT_DATE': ['01-JAN-2024'],
        'RECOVERY_STATUS': ['T'],
        'LAST_PAYMENT_DATE': ['31-DEC-2023'],
        'DUE_FREQUENCY': ['M'],
        'ASSET_TYPE_NAME': ['CARS'],
        'MAKE': ['TOYOTA'],
        'MODEL_NAME': ['COROLLA'],
        'REGISTRATION': ['REGISTERED'],
        'REGISTRATION_NO': ['TEST-001'],
        'GENDER': ['M'],
        'CITY': ['COLOMBO'],
        'DISTRICT_NAME': ['COLOMBO'],
        'PROVINCE_NAME': ['WESTERN'],
        'MARITAL_STATUS': ['S']
    })
    
    print("📝 Customer Application:")
    print(f"   Age: {customer['AGE'].iloc[0]}")
    print(f"   Income: ${customer['INCOME'].iloc[0]:,}")
    print(f"   Loan Amount: ${customer['FINANCE_AMOUNT'].iloc[0]:,}")
    print(f"   Monthly Payment: ${customer['NET_RENTAL'].iloc[0]:,}")
    print(f"   Asset: {customer['MAKE'].iloc[0]} {customer['MODEL_NAME'].iloc[0]}")
    
    # Step 4: Predict risk
    print("\n🎯 Calculating risk score...")
    try:
        result = scorer.predict(customer)
        
        # Display results
        scorer.print_risk_assessment(result)
        
        # Business decision
        print("💼 BUSINESS DECISION:")
        if result['action'] == 'APPROVE':
            print("   ✅ APPROVE: Standard loan terms")
        elif result['action'] == 'APPROVE_WITH_CONDITIONS':
            print("   ⚠️ CONDITIONAL APPROVAL:")
            print("      • Higher interest rate (+2%)")
            print("      • Require guarantor")
            print("      • Monthly monitoring")
        else:
            print("   ❌ REJECT: High default risk")
        
        return result
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_batch_prediction():
    """Test batch prediction with multiple customers"""
    
    print("\n" + "="*70)
    print("👥 TESTING BATCH PREDICTION")
    print("="*70)
    
    # Create multiple test customers
    customers = pd.DataFrame({
        # Customer IDs
        'customer_id': ['CUST001', 'CUST002', 'CUST003'],
        
        # Numerical features
        'NET_RENTAL': [8000, 15000, 20000],
        'NO_OF_RENTAL': [24, 36, 48],
        'PAID_RENTALS': [20, 30, 45],
        'CB_ARREARS_AGE': [0, 2, 5],
        'YOM': [2015, 2018, 2020],
        'FINANCE_AMOUNT': [200000, 350000, 500000],
        'CUSTOMER_VALUATION': [250000, 400000, 600000],
        'EFFECTIVE_RATE': [22.5, 25.0, 28.0],
        'AGE': [28, 40, 55],
        'INCOME': [35000, 65000, 85000],
        'EXPENSE': [12000, 20000, 25000],
        
        # Categorical features
        'PRODUCT_CODE': ['HP', 'HP', 'LN'],
        'PRODUCT_NAME': ['HIRE PURCHASE', 'HIRE PURCHASE', 'LOAN'],
        'PRODUCT_CATEGORY': ['HIRE PURCHASE', 'HIRE PURCHASE', 'PERSONAL'],
        'CONTRACT_NO': ['BATCH001', 'BATCH002', 'BATCH003'],
        'CONTRACT_STATUS': ['S', 'S', 'W'],
        'CONTRACT_DATE': ['01-JAN-2024', '15-FEB-2024', '01-MAR-2024'],
        'RECOVERY_STATUS': ['T', 'T', 'TRS'],
        'LAST_PAYMENT_DATE': ['31-DEC-2023', '28-FEB-2024', '15-MAR-2024'],
        'DUE_FREQUENCY': ['M', 'M', 'M'],
        'ASSET_TYPE_NAME': ['CARS', 'CARS', 'PERSONAL'],
        'MAKE': ['TOYOTA', 'HONDA', 'N/A'],
        'MODEL_NAME': ['COROLLA', 'CIVIC', 'N/A'],
        'REGISTRATION': ['REGISTERED', 'REGISTERED', 'N/A'],
        'REGISTRATION_NO': ['BAT-001', 'BAT-002', 'N/A'],
        'GENDER': ['M', 'F', 'M'],
        'CITY': ['COLOMBO', 'KANDY', 'GALLE'],
        'DISTRICT_NAME': ['COLOMBO', 'KANDY', 'GALLE'],
        'PROVINCE_NAME': ['WESTERN', 'CENTRAL', 'SOUTHERN'],
        'MARITAL_STATUS': ['S', 'M', 'M']
    })
    
    print(f"📊 Processing {len(customers)} applications...")
    
    try:
        # Initialize scorer
        scorer = CreditRiskScorer(
            model_path='models/autoencoder/default_autoencoder.pth',
            preprocessor_path='models/preprocessor/preprocessor.pkl'
        )
        
        # Calibrate
        val_errors = np.load('results/validation_errors.npy')
        scorer.calibrate_error_range(val_errors)
        
        # Predict batch
        results_df = scorer.predict_batch(customers)
        
        # Display results
        print("\n📋 BATCH RESULTS:")
        display_cols = ['customer_id', 'AGE', 'INCOME', 'FINANCE_AMOUNT', 
                       'risk_score', 'risk_category', 'action']
        print(results_df[display_cols].to_string(index=False))
        
        # Summary
        print(f"\n📊 SUMMARY:")
        actions = results_df['action'].value_counts()
        print(f"   Total applications: {len(results_df)}")
        for action, count in actions.items():
            print(f"   {action}: {count}")
        
        # Save results
        results_df.to_csv('results/test_batch_results.csv', index=False)
        print(f"\n💾 Results saved to: results/test_batch_results.csv")
        
        return results_df
        
    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    """Run complete system test"""
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║                  COMPLETE SYSTEM TEST                            ║
║              Credit Risk Assessment Pipeline                      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Test single prediction
    single_result = test_system()
    
    if single_result:
        # Test batch prediction
        batch_results = test_batch_prediction()
        
        if batch_results is not None:
            print("\n" + "="*70)
            print("✅ ALL TESTS PASSED!")
            print("="*70)
            print("""
🎯 Your Credit Risk System is Working!

📁 Generated Files:
   • models/autoencoder/default_autoencoder.pth
   • models/preprocessor/preprocessor.pkl
   • results/validation_errors.npy
   • results/test_batch_results.csv

💼 Ready for Production Use!
            """)
        else:
            print("\n❌ Batch test failed")
    else:
        print("\n❌ Single prediction test failed")