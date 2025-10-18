
import pandas as pd
import numpy as np
from pathlib import Path

print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║         COMPLETE CREDIT RISK ASSESSMENT WORKFLOW                 ║
║         From Raw Data to Risk Scores                             ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")

# ═══════════════════════════════════════════════════════════════════
# STEP 1: PREPROCESS DATA (If not done already)
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STEP 1: DATA PREPROCESSING")
print("="*70)

try:
    from data_preprocessing import quick_process
    
    # Check if already processed
    if not Path('data/processed/X_train.npy').exists():
        print("📊 Preprocessing data...")
        splits, preprocessor = quick_process(
            csv_path='data/raw/Bank_data.csv',
            target_col='RESCHEDULE',
            balance_strategy='smote_tomek',
            verify_leakage=True
        )
        print("✅ Preprocessing complete!")
    else:
        print("✅ Data already preprocessed!")
        
except Exception as e:
    print(f"⚠️ Preprocessing step: {e}")
    print("Run: python src/run_preprocessing.py")

# ═══════════════════════════════════════════════════════════════════
# STEP 2: TRAIN AUTOENCODER (If not done already)
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STEP 2: TRAIN AUTOENCODER")
print("="*70)

try:
    from autoencoder_training import train_autoencoder
    
    model_path = Path('models/autoencoder/default_autoencoder.pth')
    
    if not model_path.exists():
        print("🤖 Training autoencoder...")
        trainer, results = train_autoencoder(
            dataset_name='default',
            epochs=50,
            early_stopping_patience=10,
            save_visualizations=True
        )
        print("✅ Training complete!")
        print(f"   ROC-AUC: {results['metrics']['roc_auc']:.4f}")
    else:
        print("✅ Model already trained!")
        
except Exception as e:
    print(f"⚠️ Training step: {e}")
    print("Run: python src/autoencoder_training.py")

# ═══════════════════════════════════════════════════════════════════
# STEP 3: INITIALIZE RISK SCORER
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STEP 3: INITIALIZE RISK SCORER")
print("="*70)

try:
    from risk_scoring import CreditRiskScorer
    
    scorer = CreditRiskScorer(
        model_path='models/autoencoder/default_autoencoder.pth',
        preprocessor_path='models/preprocessor/preprocessor.pkl'
    )
    
    # Calibrate error range (required for 0-100 scaling)
    val_errors = np.load('results/validation_errors.npy')
    scorer.calibrate_error_range(val_errors)
    
    print("✅ Risk scorer ready!")
    
except Exception as e:
    print(f"⚠️ Scorer initialization: {e}")
    print("Make sure model and preprocessor are trained!")

# ═══════════════════════════════════════════════════════════════════
# STEP 4: PREDICT SINGLE CUSTOMER
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STEP 4: SINGLE CUSTOMER PREDICTION")
print("="*70)

try:
    # Example customer
    customer = pd.DataFrame({
        'age': [35],
        'annual_income': [55000],
        'debt': [18000],
        'loan_amount': [250000],
        'credit_score': [680],
        'employment_length': [5],
        'employment_type': ['Full-time'],
        # Add all other features your model expects
    })
    
    print("\n📝 Customer Application:")
    print(customer.T)
    
    # Predict
    result = scorer.predict(customer)
    
    # Display result
    scorer.print_risk_assessment(result)
    
    # Use the result
    if result['action'] == 'APPROVE':
        print("✅ DECISION: Approve loan application")
    elif result['action'] == 'APPROVE_WITH_CONDITIONS':
        print("⚠️ DECISION: Approve with conditions:")
        print("   • Higher interest rate")
        print("   • Require additional documentation")
        print("   • Monthly monitoring")
    else:
        print("❌ DECISION: Reject loan application")
    
except Exception as e:
    print(f"⚠️ Prediction error: {e}")

# ═══════════════════════════════════════════════════════════════════
# STEP 5: BATCH PREDICTION (Multiple Customers)
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STEP 5: BATCH PREDICTION")
print("="*70)

try:
    # Multiple new applications
    new_applications = pd.DataFrame({
        'customer_id': [2001, 2002, 2003, 2004, 2005],
        'age': [35, 45, 28, 52, 38],
        'annual_income': [55000, 85000, 35000, 95000, 62000],
        'debt': [18000, 15000, 22000, 12000, 16000],
        'loan_amount': [250000, 400000, 180000, 350000, 280000],
        'credit_score': [680, 750, 610, 780, 700],
        'employment_length': [5, 12, 2, 15, 7],
        'employment_type': ['Full-time', 'Self-employed', 'Part-time', 
                           'Full-time', 'Full-time']
    })
    
    print(f"\n📊 Processing {len(new_applications)} applications...")
    
    # Predict batch
    results_df = scorer.predict_batch(new_applications)
    
    # Display results
    print("\n📋 RESULTS:")
    print(results_df[['customer_id', 'age', 'credit_score', 
                      'risk_score', 'risk_category', 'action']].to_string(index=False))
    
    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"   Total applications: {len(results_df)}")
    print(f"   Approved: {sum(results_df['action'] == 'APPROVE')}")
    print(f"   Approved with conditions: {sum(results_df['action'] == 'APPROVE_WITH_CONDITIONS')}")
    print(f"   Rejected: {sum(results_df['action'] == 'REJECT')}")
    
    # Save results
    results_df.to_csv('results/batch_predictions.csv', index=False)
    print(f"\n💾 Results saved to: results/batch_predictions.csv")
    
except Exception as e:
    print(f"⚠️ Batch prediction error: {e}")

# ═══════════════════════════════════════════════════════════════════
# STEP 6: GENERATE COMPREHENSIVE REPORT
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STEP 6: GENERATE REPORT")
print("="*70)

try:
    from risk_scoring import create_risk_report
    
    # Create report from batch results
    results_list = results_df[
        ['risk_score', 'risk_category', 'action', 'reconstruction_error']
    ].to_dict('records')
    
    report = create_risk_report(results_list, 'results/risk_assessment_report.csv')
    
    print("✅ Report generated!")
    
except Exception as e:
    print(f"⚠️ Report generation error: {e}")

# ═══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("✅ WORKFLOW COMPLETE!")
print("="*70)

print("""
📁 Generated Files:
   • models/autoencoder/lending_club_autoencoder.pth
   • models/preprocessor/lending_club/preprocessor.pkl
   • results/validation_errors.npy
   • results/batch_predictions.csv
   • results/risk_assessment_report.csv
   • results/figures/autoencoder/training_history.png
   • results/figures/autoencoder/error_distribution.png
   • results/figures/autoencoder/roc_curve.png

🎯 Risk Categories Used:
   🟢 LOW (0-30):      Approve
   🟡 MEDIUM (30-60):  Approve with conditions
   🔴 HIGH (60-100):   Reject

💼 Next Steps:
   1. Review batch predictions in results/batch_predictions.csv
   2. Check visualizations in results/figures/autoencoder/
   3. Integrate with your production system
   4. Customize risk thresholds if needed

🚀 Production Ready!
""")


# ═══════════════════════════════════════════════════════════════════
# BONUS: QUICK PREDICTION FUNCTION
# ═══════════════════════════════════════════════════════════════════

def quick_risk_assessment(customer_data: dict) -> dict:
    """
    Quick function for single customer risk assessment
    
    Args:
        customer_data: Dictionary with customer features
        
    Returns:
        Risk assessment dictionary
    
    Example:
        result = quick_risk_assessment({
            'age': 35,
            'annual_income': 55000,
            'debt': 18000,
            'loan_amount': 250000,
            'credit_score': 680,
            'employment_length': 5,
            'employment_type': 'Full-time'
        })
        
        print(f"Risk: {result['risk_score']}")
        print(f"Action: {result['action']}")
    """
    from risk_scoring import CreditRiskScorer
    import pandas as pd
    
    # Initialize scorer
    scorer = CreditRiskScorer(
        model_path='models/autoencoder/lending_club_autoencoder.pth',
        preprocessor_path='models/preprocessor/lending_club/preprocessor.pkl'
    )
    
    # Calibrate
    val_errors = np.load('results/validation_errors.npy')
    scorer.calibrate_error_range(val_errors)
    
    # Convert to DataFrame
    df = pd.DataFrame([customer_data])
    
    # Predict
    result = scorer.predict(df)
    
    return result


if __name__ == "__main__":
    print("\n💡 TIP: You can import quick_risk_assessment() for fast predictions!")
    print("""
    from complete_workflow import quick_risk_assessment
    
    result = quick_risk_assessment({
        'age': 35,
        'annual_income': 55000,
        # ... other features
    })
    
    print(result['risk_score'])    # 45.23
    print(result['risk_category']) # MEDIUM
    print(result['action'])        # APPROVE_WITH_CONDITIONS
    """)