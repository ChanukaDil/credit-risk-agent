#!/usr/bin/env python3
"""
🎯 QUICK MODEL TESTER
Test your credit risk model with sample values in seconds!
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

def quick_test():
    """Quick test with sample values"""
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║                    🎯 QUICK MODEL TEST                           ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        from risk_scoring import CreditRiskScorer
        
        # Initialize model
        print("📂 Loading your trained model...")
        scorer = CreditRiskScorer(
            model_path='models/autoencoder/default_autoencoder.pth',
            preprocessor_path='models/preprocessor/preprocessor.pkl'
        )
        
        # Calibrate
        val_errors = np.load('results/validation_errors.npy')
        scorer.calibrate_error_range(val_errors)
        print("✅ Model loaded and calibrated!")
        
        # Sample customer - YOU CAN EDIT THESE VALUES! 
        print(f"\n🏗️ Creating sample customer...")
        print(f"   (You can edit the values below in the script)")
        
        # ========================================
        # 🔧 EDIT THESE VALUES TO TEST YOUR MODEL
        # ========================================
        sample_customer = pd.DataFrame({
            # Basic Info
            'NET_RENTAL': [10000],       # Monthly payment
            'NO_OF_RENTAL': [48],        # Total payments
            'PAID_RENTALS': [0],         # Payments made
            'CB_ARREARS_AGE': [0],       # Days overdue
            'YOM': [2020],               # Vehicle year
            'FINANCE_AMOUNT': [300000],  # Loan amount  
            'CUSTOMER_VALUATION': [350000], # Asset value
            'EFFECTIVE_RATE': [16.5],    # Interest rate
            'AGE': [32],                 # Customer age
            'INCOME': [55000],           # Annual income
            'EXPENSE': [22000],          # Annual expenses
            
            # Categories (use these exact values)
            'PRODUCT_CODE': ['HP'],
            'PRODUCT_NAME': ['HIRE PURCHASE'],
            'PRODUCT_CATEGORY': ['HIRE PURCHASE'],
            'CONTRACT_NO': ['TEST123'],
            'CONTRACT_STATUS': ['S'],    # S=Active
            'CONTRACT_DATE': ['15-OCT-2024'],
            'RECOVERY_STATUS': ['T'],    # T=Good
            'LAST_PAYMENT_DATE': ['15-SEP-2024'],
            'DUE_FREQUENCY': ['M'],      # M=Monthly
            'ASSET_TYPE_NAME': ['CARS'],
            'MAKE': ['TOYOTA'],
            'MODEL_NAME': ['COROLLA'],
            'REGISTRATION': ['REGISTERED'],
            'REGISTRATION_NO': ['ABC-1234'],
            'GENDER': ['M'],             # M or F
            'CITY': ['COLOMBO'],
            'DISTRICT_NAME': ['COLOMBO'],
            'PROVINCE_NAME': ['WESTERN'],
            'MARITAL_STATUS': ['M']      # M=Married, S=Single
        })
        
        print(f"\n📝 Sample Customer Profile:")
        print(f"   👤 Age: {sample_customer['AGE'].iloc[0]}")
        print(f"   💰 Income: ${sample_customer['INCOME'].iloc[0]:,}")
        print(f"   🏦 Loan Amount: ${sample_customer['FINANCE_AMOUNT'].iloc[0]:,}")
        print(f"   💳 Monthly Payment: ${sample_customer['NET_RENTAL'].iloc[0]:,}")
        print(f"   🚗 Vehicle: {sample_customer['YOM'].iloc[0]} {sample_customer['MAKE'].iloc[0]} {sample_customer['MODEL_NAME'].iloc[0]}")
        
        # Make prediction
        print(f"\n🎯 Making prediction...")
        result = scorer.predict(sample_customer)
        
        # Show detailed results
        scorer.print_risk_assessment(result)
        
        # Business interpretation
        print(f"💼 BUSINESS RECOMMENDATION:")
        if result['risk_score'] < 30:
            print(f"   ✅ APPROVE with standard terms")
            print(f"   📊 Low risk customer")
        elif result['risk_score'] < 60:
            print(f"   ⚠️ APPROVE with conditions:")
            print(f"   • Higher interest rate (+2-3%)")
            print(f"   • Require guarantor or additional documentation")
        else:
            print(f"   ❌ REJECT or require significant conditions:")
            print(f"   • High default risk")
            print(f"   • Consider larger down payment or co-signer")
        
        return result
        
    except ImportError:
        print(f"❌ Could not import risk_scoring module")
        print(f"   Make sure you're running from the project root directory")
        return None
    except FileNotFoundError as e:
        print(f"❌ Required file not found: {e}")
        print(f"\n🔧 Make sure you have:")
        print(f"   1. Trained the model: python src/autoencoder_training.py")
        print(f"   2. Processed data: python src/run_preprocessing.py")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_multiple_scenarios():
    """Test different risk scenarios"""
    
    print(f"\n{'='*60}")
    print(f"🧪 TESTING MULTIPLE RISK SCENARIOS")
    print(f"{'='*60}")
    
    scenarios = [
        {
            'name': '🟢 LOW RISK',
            'values': {
                'NET_RENTAL': 8000, 'FINANCE_AMOUNT': 200000, 'AGE': 35, 
                'INCOME': 60000, 'EXPENSE': 20000, 'CB_ARREARS_AGE': 0
            }
        },
        {
            'name': '🟡 MEDIUM RISK', 
            'values': {
                'NET_RENTAL': 12000, 'FINANCE_AMOUNT': 350000, 'AGE': 25,
                'INCOME': 40000, 'EXPENSE': 30000, 'CB_ARREARS_AGE': 30
            }
        },
        {
            'name': '🔴 HIGH RISK',
            'values': {
                'NET_RENTAL': 15000, 'FINANCE_AMOUNT': 500000, 'AGE': 22,
                'INCOME': 30000, 'EXPENSE': 28000, 'CB_ARREARS_AGE': 90
            }
        }
    ]
    
    try:
        from risk_scoring import CreditRiskScorer
        
        scorer = CreditRiskScorer(
            model_path='models/autoencoder/default_autoencoder.pth',
            preprocessor_path='models/preprocessor/preprocessor.pkl'
        )
        val_errors = np.load('results/validation_errors.npy')
        scorer.calibrate_error_range(val_errors)
        
        results = []
        
        for scenario in scenarios:
            print(f"\n{scenario['name']} Customer:")
            
            # Base template
            customer_data = {
                'NET_RENTAL': [scenario['values']['NET_RENTAL']],
                'NO_OF_RENTAL': [48], 'PAID_RENTALS': [0],
                'CB_ARREARS_AGE': [scenario['values']['CB_ARREARS_AGE']],
                'YOM': [2018], 'FINANCE_AMOUNT': [scenario['values']['FINANCE_AMOUNT']],
                'CUSTOMER_VALUATION': [scenario['values']['FINANCE_AMOUNT'] * 1.2],
                'EFFECTIVE_RATE': [16.5], 'AGE': [scenario['values']['AGE']],
                'INCOME': [scenario['values']['INCOME']], 
                'EXPENSE': [scenario['values']['EXPENSE']],
                'PRODUCT_CODE': ['HP'], 'PRODUCT_NAME': ['HIRE PURCHASE'],
                'PRODUCT_CATEGORY': ['HIRE PURCHASE'], 'CONTRACT_NO': [f'TEST{len(results)+1}'],
                'CONTRACT_STATUS': ['S'], 'CONTRACT_DATE': ['15-OCT-2024'],
                'RECOVERY_STATUS': ['T'], 'LAST_PAYMENT_DATE': ['15-SEP-2024'],
                'DUE_FREQUENCY': ['M'], 'ASSET_TYPE_NAME': ['CARS'],
                'MAKE': ['TOYOTA'], 'MODEL_NAME': ['COROLLA'],
                'REGISTRATION': ['REGISTERED'], 'REGISTRATION_NO': [f'ABC-{len(results)+1}'],
                'GENDER': ['M'], 'CITY': ['COLOMBO'], 'DISTRICT_NAME': ['COLOMBO'],
                'PROVINCE_NAME': ['WESTERN'], 'MARITAL_STATUS': ['M']
            }
            
            customer_df = pd.DataFrame(customer_data)
            result = scorer.predict(customer_df)
            
            # Show summary
            print(f"   💰 Income: ${scenario['values']['INCOME']:,}")
            print(f"   🏦 Loan: ${scenario['values']['FINANCE_AMOUNT']:,}")
            print(f"   📊 Risk Score: {result['risk_score']:.1f}/100")
            print(f"   🎯 Category: {result['risk_category']}")
            print(f"   💼 Action: {result['action']}")
            
            results.append(result)
        
        print(f"\n📊 SUMMARY COMPARISON:")
        print(f"   🟢 Low Risk Score:    {results[0]['risk_score']:.1f}")
        print(f"   🟡 Medium Risk Score: {results[1]['risk_score']:.1f}")  
        print(f"   🔴 High Risk Score:   {results[2]['risk_score']:.1f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    """Run the quick test"""
    
    print("🚀 Starting quick model test...")
    
    # Test 1: Single sample
    result = quick_test()
    
    if result:
        print(f"\n✅ Single test completed!")
        
        # Test 2: Multiple scenarios
        scenario_results = test_multiple_scenarios()
        
        if scenario_results:
            print(f"\n✅ All tests completed!")
            print(f"\n🎯 YOUR MODEL IS WORKING!")
            print(f"\n💡 How to customize:")
            print(f"   1. Edit the values in sample_customer above")
            print(f"   2. Run this script again: python quick_test.py")
            print(f"   3. Try different income, loan amounts, ages, etc.")
            
            print(f"\n📖 Understanding the results:")
            print(f"   • Risk Score 0-30:  Low risk → Approve")
            print(f"   • Risk Score 30-60: Medium risk → Approve with conditions") 
            print(f"   • Risk Score 60-100: High risk → Reject")
        else:
            print(f"⚠️ Scenario tests failed")
    else:
        print(f"❌ Quick test failed - check if model is trained")
        print(f"\n🔧 To train the model:")
        print(f"   1. python src/run_preprocessing.py")
        print(f"   2. python src/autoencoder_training.py")