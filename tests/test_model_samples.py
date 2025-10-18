#!/usr/bin/env python3
"""
🧪 MODEL TESTING SCRIPT
Test your Credit Risk Model with Sample Values

This script shows you exactly how to:
1. Create sample customer data
2. Test individual predictions
3. Test batch predictions
4. Understand the results
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append('src')

from risk_scoring import CreditRiskScorer

def create_sample_customer(risk_profile='medium'):
    """
    Create sample customer data with the exact features your model expects
    
    Args:
        risk_profile: 'low', 'medium', 'high' - creates different risk scenarios
    
    Returns:
        DataFrame with one customer
    """
    
    print(f"\n🏗️ Creating {risk_profile.upper()} RISK customer sample...")
    
    if risk_profile == 'low':
        # Good customer profile
        customer = {
            # Basic loan info
            'NET_RENTAL': 8000,          # Lower monthly payment
            'NO_OF_RENTAL': 36,          # Shorter term
            'PAID_RENTALS': 0,           # New customer
            'CB_ARREARS_AGE': 0,         # No arrears
            'YOM': 2018,                 # Newer vehicle
            'FINANCE_AMOUNT': 200000,    # Lower loan amount
            'CUSTOMER_VALUATION': 250000, # Higher asset value
            'EFFECTIVE_RATE': 15.5,      # Standard rate
            'AGE': 35,                   # Stable age
            'INCOME': 60000,             # Good income
            'EXPENSE': 25000,            # Reasonable expenses
            
            # Categorical features (will be encoded)
            'PRODUCT_CODE': 'HP',
            'PRODUCT_NAME': 'HIRE PURCHASE',
            'PRODUCT_CATEGORY': 'HIRE PURCHASE',
            'CONTRACT_NO': 'TEST001',
            'CONTRACT_STATUS': 'S',      # Active
            'CONTRACT_DATE': '15-OCT-2024',
            'RECOVERY_STATUS': 'T',      # Good status
            'LAST_PAYMENT_DATE': '15-SEP-2024',
            'DUE_FREQUENCY': 'M',        # Monthly
            'ASSET_TYPE_NAME': 'CARS',
            'MAKE': 'TOYOTA',
            'MODEL_NAME': 'COROLLA',
            'REGISTRATION': 'REGISTERED',
            'REGISTRATION_NO': 'ABC-1234',
            'GENDER': 'M',
            'CITY': 'COLOMBO',
            'DISTRICT_NAME': 'COLOMBO',
            'PROVINCE_NAME': 'WESTERN',
            'MARITAL_STATUS': 'M'        # Married
        }
        
    elif risk_profile == 'medium':
        # Medium risk customer
        customer = {
            'NET_RENTAL': 12000,         # Higher payment
            'NO_OF_RENTAL': 60,          # Longer term
            'PAID_RENTALS': 5,           # Some payments made
            'CB_ARREARS_AGE': 30,        # Minor arrears
            'YOM': 2015,                 # Older vehicle
            'FINANCE_AMOUNT': 350000,    # Higher loan
            'CUSTOMER_VALUATION': 400000,
            'EFFECTIVE_RATE': 18.0,      # Higher rate
            'AGE': 28,                   # Younger
            'INCOME': 45000,             # Lower income
            'EXPENSE': 30000,            # Higher expenses
            
            'PRODUCT_CODE': 'HP',
            'PRODUCT_NAME': 'HIRE PURCHASE',
            'PRODUCT_CATEGORY': 'HIRE PURCHASE',
            'CONTRACT_NO': 'TEST002',
            'CONTRACT_STATUS': 'S',
            'CONTRACT_DATE': '01-AUG-2024',
            'RECOVERY_STATUS': 'T',
            'LAST_PAYMENT_DATE': '01-OCT-2024',
            'DUE_FREQUENCY': 'M',
            'ASSET_TYPE_NAME': 'CARS',
            'MAKE': 'NISSAN',
            'MODEL_NAME': 'SUNNY',
            'REGISTRATION': 'UNREGISTERED',
            'REGISTRATION_NO': 'TEMP-5678',
            'GENDER': 'F',
            'CITY': 'KANDY',
            'DISTRICT_NAME': 'KANDY',
            'PROVINCE_NAME': 'CENTRAL',
            'MARITAL_STATUS': 'S'        # Single
        }
        
    elif risk_profile == 'high':
        # High risk customer
        customer = {
            'NET_RENTAL': 15000,         # Very high payment
            'NO_OF_RENTAL': 72,          # Very long term
            'PAID_RENTALS': 12,          # Many payments behind
            'CB_ARREARS_AGE': 90,        # Significant arrears
            'YOM': 2010,                 # Old vehicle
            'FINANCE_AMOUNT': 500000,    # Very high loan
            'CUSTOMER_VALUATION': 450000, # Lower asset value vs loan
            'EFFECTIVE_RATE': 22.0,      # High risk rate
            'AGE': 22,                   # Very young
            'INCOME': 30000,             # Low income
            'EXPENSE': 28000,            # High expense ratio
            
            'PRODUCT_CODE': 'HP',
            'PRODUCT_NAME': 'HIRE PURCHASE',
            'PRODUCT_CATEGORY': 'HIRE PURCHASE',
            'CONTRACT_NO': 'TEST003',
            'CONTRACT_STATUS': 'W',      # Problematic status
            'CONTRACT_DATE': '01-JAN-2023',
            'RECOVERY_STATUS': 'TRS',    # Recovery status
            'LAST_PAYMENT_DATE': '15-JUL-2024',
            'DUE_FREQUENCY': 'M',
            'ASSET_TYPE_NAME': 'MOTOR CYCLES',
            'MAKE': 'HONDA',
            'MODEL_NAME': 'CD125',
            'REGISTRATION': 'UNREGISTERED',
            'REGISTRATION_NO': 'NONE',
            'GENDER': 'M',
            'CITY': 'GALLE',
            'DISTRICT_NAME': 'GALLE',
            'PROVINCE_NAME': 'SOUTHERN',
            'MARITAL_STATUS': 'S'
        }
    
    # Convert to DataFrame
    df = pd.DataFrame([customer])
    
    print(f"✅ Sample customer created:")
    print(f"   👤 Age: {customer['AGE']}")
    print(f"   💰 Income: ${customer['INCOME']:,}")
    print(f"   🏦 Loan: ${customer['FINANCE_AMOUNT']:,}")
    print(f"   💳 Monthly Payment: ${customer['NET_RENTAL']:,}")
    print(f"   🚗 Vehicle: {customer['YOM']} {customer['MAKE']} {customer['MODEL_NAME']}")
    
    return df

def test_single_prediction(risk_profile='medium'):
    """Test prediction for a single customer"""
    
    print(f"\n{'='*80}")
    print(f"🧪 TESTING SINGLE CUSTOMER PREDICTION ({risk_profile.upper()} RISK)")
    print(f"{'='*80}")
    
    try:
        # Step 1: Initialize the scorer
        print("\n📂 Loading trained model...")
        scorer = CreditRiskScorer(
            model_path='models/autoencoder/default_autoencoder.pth',
            preprocessor_path='models/preprocessor/preprocessor.pkl'
        )
        
        # Step 2: Calibrate risk scoring
        print("⚖️ Calibrating risk scoring...")
        val_errors = np.load('results/validation_errors.npy')
        scorer.calibrate_error_range(val_errors)
        
        # Step 3: Create sample customer
        customer = create_sample_customer(risk_profile)
        
        # Step 4: Make prediction
        print("\n🎯 Making prediction...")
        result = scorer.predict(customer)
        
        # Step 5: Display results
        scorer.print_risk_assessment(result)
        
        # Step 6: Business interpretation
        print(f"\n💼 BUSINESS DECISION:")
        if result['action'] == 'APPROVE':
            print(f"   ✅ APPROVE: Standard loan terms")
            print(f"   📊 Risk Score: {result['risk_score']:.1f}/100 (LOW RISK)")
            print(f"   💰 Recommended: Standard interest rate")
        elif result['action'] == 'APPROVE_WITH_CONDITIONS':
            print(f"   ⚠️ APPROVE WITH CONDITIONS:")
            print(f"   📊 Risk Score: {result['risk_score']:.1f}/100 (MEDIUM RISK)")
            print(f"   💰 Recommended: Higher interest rate (+2-3%)")
            print(f"   📄 Required: Additional documentation, guarantor")
            print(f"   📅 Monitoring: Monthly payment reviews")
        else:
            print(f"   ❌ REJECT:")
            print(f"   📊 Risk Score: {result['risk_score']:.1f}/100 (HIGH RISK)")
            print(f"   🚫 Reason: Too high default probability")
            print(f"   💡 Suggestion: Require larger down payment or co-signer")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"\n🔧 Make sure you have:")
        print(f"   • Trained the model: python src/autoencoder_training.py")
        print(f"   • Processed the data: python src/run_preprocessing.py")
        return None

def test_batch_predictions():
    """Test predictions for multiple customers"""
    
    print(f"\n{'='*80}")
    print(f"👥 TESTING BATCH PREDICTIONS (Multiple Customers)")
    print(f"{'='*80}")
    
    try:
        # Initialize scorer
        scorer = CreditRiskScorer(
            model_path='models/autoencoder/default_autoencoder.pth',
            preprocessor_path='models/preprocessor/preprocessor.pkl'
        )
        
        val_errors = np.load('results/validation_errors.npy')
        scorer.calibrate_error_range(val_errors)
        
        # Create multiple customers
        print(f"\n🏗️ Creating batch of customers...")
        
        customers = []
        for risk_profile in ['low', 'medium', 'high']:
            customer = create_sample_customer(risk_profile)
            customers.append(customer)
        
        # Combine into one DataFrame
        batch_df = pd.concat(customers, ignore_index=True)
        
        # Add customer IDs
        batch_df['CUSTOMER_ID'] = ['CUST_LOW_001', 'CUST_MED_002', 'CUST_HIGH_003']
        
        print(f"\n🎯 Making batch predictions...")
        results_df = scorer.predict_batch(batch_df)
        
        # Display results
        print(f"\n📋 BATCH RESULTS:")
        print("="*120)
        
        display_cols = ['CUSTOMER_ID', 'AGE', 'INCOME', 'FINANCE_AMOUNT', 'NET_RENTAL', 
                       'risk_score', 'risk_category', 'action']
        print(results_df[display_cols].to_string(index=False))
        
        # Summary statistics
        print(f"\n📊 BATCH SUMMARY:")
        print("="*50)
        total = len(results_df)
        approved = sum(results_df['action'] == 'APPROVE')
        conditional = sum(results_df['action'] == 'APPROVE_WITH_CONDITIONS')
        rejected = sum(results_df['action'] == 'REJECT')
        
        print(f"📈 Total Applications: {total}")
        print(f"✅ Approved: {approved} ({approved/total*100:.1f}%)")
        print(f"⚠️ Approved with Conditions: {conditional} ({conditional/total*100:.1f}%)")
        print(f"❌ Rejected: {rejected} ({rejected/total*100:.1f}%)")
        
        print(f"\n💰 Risk Score Statistics:")
        print(f"   Average: {results_df['risk_score'].mean():.1f}")
        print(f"   Lowest:  {results_df['risk_score'].min():.1f}")
        print(f"   Highest: {results_df['risk_score'].max():.1f}")
        
        # Save results
        output_path = 'results/model_test_results.csv'
        results_df.to_csv(output_path, index=False)
        print(f"\n💾 Results saved to: {output_path}")
        
        return results_df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_custom_customer():
    """Test with your own custom values"""
    
    print(f"\n{'='*80}")
    print(f"✏️ TESTING CUSTOM CUSTOMER (Edit the values below)")
    print(f"{'='*80}")
    
    # 🔧 EDIT THESE VALUES TO TEST YOUR OWN CUSTOMER
    custom_values = {
        'NET_RENTAL': 10000,         # Monthly payment amount
        'NO_OF_RENTAL': 48,          # Total number of payments
        'PAID_RENTALS': 2,           # Payments already made
        'CB_ARREARS_AGE': 0,         # Days in arrears (0 = no arrears)
        'YOM': 2020,                 # Year of manufacture
        'FINANCE_AMOUNT': 300000,    # Loan amount
        'CUSTOMER_VALUATION': 350000, # Asset value
        'EFFECTIVE_RATE': 16.5,      # Interest rate %
        'AGE': 30,                   # Customer age
        'INCOME': 50000,             # Annual income
        'EXPENSE': 20000,            # Annual expenses
        
        # Categorical values (choose from options below)
        'PRODUCT_CODE': 'HP',        # HP, LB, etc.
        'PRODUCT_NAME': 'HIRE PURCHASE',
        'PRODUCT_CATEGORY': 'HIRE PURCHASE',
        'CONTRACT_NO': 'CUSTOM001',
        'CONTRACT_STATUS': 'S',      # S=Active, W=Problem
        'CONTRACT_DATE': '15-OCT-2024',
        'RECOVERY_STATUS': 'T',      # T=Good, TRS=Recovery
        'LAST_PAYMENT_DATE': '15-SEP-2024',
        'DUE_FREQUENCY': 'M',        # M=Monthly
        'ASSET_TYPE_NAME': 'CARS',   # CARS, MOTOR CYCLES
        'MAKE': 'TOYOTA',            # TOYOTA, NISSAN, HONDA, etc.
        'MODEL_NAME': 'CAMRY',       # COROLLA, SUNNY, etc.
        'REGISTRATION': 'REGISTERED', # REGISTERED, UNREGISTERED
        'REGISTRATION_NO': 'XYZ-9999',
        'GENDER': 'M',               # M, F
        'CITY': 'COLOMBO',
        'DISTRICT_NAME': 'COLOMBO',
        'PROVINCE_NAME': 'WESTERN',  # WESTERN, CENTRAL, SOUTHERN, etc.
        'MARITAL_STATUS': 'M'        # M=Married, S=Single
    }
    
    print(f"\n🔧 Your Custom Customer Profile:")
    print(f"   👤 Age: {custom_values['AGE']}")
    print(f"   💰 Income: ${custom_values['INCOME']:,}")
    print(f"   🏦 Loan: ${custom_values['FINANCE_AMOUNT']:,}")
    print(f"   💳 Monthly Payment: ${custom_values['NET_RENTAL']:,}")
    print(f"   🚗 Vehicle: {custom_values['YOM']} {custom_values['MAKE']} {custom_values['MODEL_NAME']}")
    print(f"   📍 Location: {custom_values['CITY']}, {custom_values['PROVINCE_NAME']}")
    
    # Convert to DataFrame and test
    custom_df = pd.DataFrame([custom_values])
    
    try:
        # Initialize and test
        scorer = CreditRiskScorer(
            model_path='models/autoencoder/default_autoencoder.pth',
            preprocessor_path='models/preprocessor/preprocessor.pkl'
        )
        
        val_errors = np.load('results/validation_errors.npy')
        scorer.calibrate_error_range(val_errors)
        
        print(f"\n🎯 Testing your custom customer...")
        result = scorer.predict(custom_df)
        
        scorer.print_risk_assessment(result)
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def show_feature_importance():
    """Show which features are most important"""
    
    print(f"\n{'='*80}")
    print(f"📊 UNDERSTANDING YOUR MODEL'S FEATURES")
    print(f"{'='*80}")
    
    print(f"\n🔍 Key Features Your Model Uses:")
    print(f"")
    print(f"💰 FINANCIAL FACTORS:")
    print(f"   • NET_RENTAL: Monthly payment amount")
    print(f"   • FINANCE_AMOUNT: Total loan amount")
    print(f"   • INCOME: Customer annual income")
    print(f"   • EXPENSE: Customer annual expenses")
    print(f"   • EFFECTIVE_RATE: Interest rate")
    print(f"   • CUSTOMER_VALUATION: Asset value")
    print(f"")
    print(f"📅 PAYMENT HISTORY:")
    print(f"   • PAID_RENTALS: Number of payments made")
    print(f"   • CB_ARREARS_AGE: Days overdue")
    print(f"   • RECOVERY_STATUS: Current payment status")
    print(f"")
    print(f"👤 CUSTOMER PROFILE:")
    print(f"   • AGE: Customer age")
    print(f"   • MARITAL_STATUS: Married/Single")
    print(f"   • GENDER: M/F")
    print(f"")
    print(f"🚗 ASSET DETAILS:")
    print(f"   • YOM: Year of manufacture")
    print(f"   • MAKE: Vehicle brand")
    print(f"   • MODEL_NAME: Vehicle model")
    print(f"   • ASSET_TYPE_NAME: Cars/Motorcycles")
    print(f"")
    print(f"📍 LOCATION:")
    print(f"   • CITY, DISTRICT_NAME, PROVINCE_NAME")
    print(f"")
    print(f"💡 TIP: The model looks for unusual patterns compared to 'normal' customers")

if __name__ == "__main__":
    """Main execution - run all tests"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    🧪 CREDIT RISK MODEL TESTING SUITE                       ║
║                                                                              ║
║     Test your trained model with different customer scenarios                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"\n🔍 Checking if model is trained...")
    
    # Check if model exists
    model_path = Path('models/autoencoder/default_autoencoder.pth')
    preprocessor_path = Path('models/preprocessor/preprocessor.pkl')
    validation_errors_path = Path('results/validation_errors.npy')
    
    if not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        print(f"🔧 Please run: python src/autoencoder_training.py")
        exit(1)
        
    if not preprocessor_path.exists():
        print(f"❌ Preprocessor not found at: {preprocessor_path}")
        print(f"🔧 Please run: python src/run_preprocessing.py")
        exit(1)
        
    if not validation_errors_path.exists():
        print(f"❌ Validation errors not found at: {validation_errors_path}")
        print(f"🔧 Please run: python src/autoencoder_training.py")
        exit(1)
    
    print(f"✅ All required files found!")
    
    # Show what features the model uses
    show_feature_importance()
    
    # Test 1: Low risk customer
    print(f"\n🟢 TEST 1: Low Risk Customer")
    result_low = test_single_prediction('low')
    
    # Test 2: Medium risk customer  
    print(f"\n🟡 TEST 2: Medium Risk Customer")
    result_medium = test_single_prediction('medium')
    
    # Test 3: High risk customer
    print(f"\n🔴 TEST 3: High Risk Customer")
    result_high = test_single_prediction('high')
    
    # Test 4: Batch prediction
    batch_results = test_batch_predictions()
    
    # Test 5: Custom customer (you can edit the values)
    print(f"\n✏️ TEST 4: Custom Customer (Edit values in the code)")
    custom_result = test_custom_customer()
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"✅ ALL TESTS COMPLETED!")
    print(f"{'='*80}")
    
    if all(r is not None for r in [result_low, result_medium, result_high]):
        print(f"\n📊 RISK SCORE COMPARISON:")
        print(f"   🟢 Low Risk Customer:    {result_low['risk_score']:.1f}/100")
        print(f"   🟡 Medium Risk Customer: {result_medium['risk_score']:.1f}/100")
        print(f"   🔴 High Risk Customer:   {result_high['risk_score']:.1f}/100")
        
        print(f"\n🎯 MODEL BEHAVIOR:")
        print(f"   • Lower scores = Lower risk = More likely to approve")
        print(f"   • Higher scores = Higher risk = More likely to reject")
        print(f"   • The model learned to distinguish risk patterns!")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Edit custom_values in test_custom_customer() to test your own scenarios")
    print(f"   2. Run this script anytime: python test_model_samples.py")
    print(f"   3. Check results/model_test_results.csv for batch results")
    print(f"   4. Use the risk scores for business decisions!")
    
    print(f"\n💡 Pro Tips:")
    print(f"   • Risk scores 0-30: Safe to approve")
    print(f"   • Risk scores 30-60: Approve with conditions (higher rate)")
    print(f"   • Risk scores 60-100: Consider rejecting")
    print(f"   • The model detects unusual patterns vs normal customers")