"""
Advanced Feature Engineering for Credit Risk
Creates domain-specific features to improve model performance
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle
import json


class CreditRiskFeatureEngineer:
    """
    Creates advanced features for credit risk prediction
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        self.original_features = []
    
    def create_financial_ratios(self, df):
        """Create financial health indicators"""
        features = {}
        
        # 1. Debt-to-Income Ratio
        if 'FINANCE_AMOUNT' in df.columns and 'INCOME' in df.columns:
            features['debt_to_income_ratio'] = df['FINANCE_AMOUNT'] / (df['INCOME'] + 1)
        
        # 2. Payment-to-Income Ratio
        if 'NET_RENTAL' in df.columns and 'INCOME' in df.columns:
            features['payment_to_income_ratio'] = df['NET_RENTAL'] / (df['INCOME'] + 1)
        
        # 3. Expense-to-Income Ratio
        if 'EXPENSE' in df.columns and 'INCOME' in df.columns:
            features['expense_to_income_ratio'] = df['EXPENSE'] / (df['INCOME'] + 1)
        
        # 4. Free Cash Flow
        if 'INCOME' in df.columns and 'EXPENSE' in df.columns and 'NET_RENTAL' in df.columns:
            features['free_cash_flow'] = df['INCOME'] - df['EXPENSE'] - df['NET_RENTAL']
        
        # 5. Loan-to-Value Ratio
        if 'FINANCE_AMOUNT' in df.columns and 'CUSTOMER_VALUATION' in df.columns:
            features['loan_to_value_ratio'] = df['FINANCE_AMOUNT'] / (df['CUSTOMER_VALUATION'] + 1)
        
        # 6. Savings Capacity
        if 'INCOME' in df.columns and 'EXPENSE' in df.columns:
            features['savings_capacity'] = (df['INCOME'] - df['EXPENSE']) / (df['INCOME'] + 1)
        
        return pd.DataFrame(features)
    
    def create_payment_features(self, df):
        """Create payment behavior features"""
        features = {}
        
        # 1. Payment Completion Rate
        if 'PAID_RENTALS' in df.columns and 'NO_OF_RENTAL' in df.columns:
            features['payment_completion_rate'] = df['PAID_RENTALS'] / (df['NO_OF_RENTAL'] + 1)
        
        # 2. Remaining Payments
        if 'NO_OF_RENTAL' in df.columns and 'PAID_RENTALS' in df.columns:
            features['remaining_payments'] = df['NO_OF_RENTAL'] - df['PAID_RENTALS']
        
        # 3. Payment Progress Score
        if 'PAID_RENTALS' in df.columns and 'NO_OF_RENTAL' in df.columns:
            features['payment_progress_score'] = (df['PAID_RENTALS'] / (df['NO_OF_RENTAL'] + 1)) * 100
        
        # 4. Average Payment Amount
        if 'NET_RENTAL' in df.columns and 'NO_OF_RENTAL' in df.columns:
            features['avg_payment_amount'] = df['NET_RENTAL'] / (df['NO_OF_RENTAL'] + 1)
        
        # 5. Total Paid Amount
        if 'NET_RENTAL' in df.columns and 'PAID_RENTALS' in df.columns:
            features['total_paid_amount'] = df['NET_RENTAL'] * df['PAID_RENTALS']
        
        # 6. Outstanding Balance
        if 'FINANCE_AMOUNT' in df.columns and 'NET_RENTAL' in df.columns and 'PAID_RENTALS' in df.columns:
            features['outstanding_balance'] = df['FINANCE_AMOUNT'] - (df['NET_RENTAL'] * df['PAID_RENTALS'])
        
        return pd.DataFrame(features)
    
    def create_risk_indicators(self, df):
        """Create risk indicator features"""
        features = {}
        
        # 1. Credit Bureau Risk Score
        if 'CB_ARREARS_AGE' in df.columns:
            features['cb_risk_high'] = (df['CB_ARREARS_AGE'] > 90).astype(int)
            features['cb_risk_medium'] = ((df['CB_ARREARS_AGE'] > 30) & (df['CB_ARREARS_AGE'] <= 90)).astype(int)
            features['cb_risk_low'] = (df['CB_ARREARS_AGE'] <= 30).astype(int)
        
        # 2. Age Risk (younger = higher risk typically)
        if 'AGE' in df.columns:
            features['age_risk_young'] = (df['AGE'] < 25).astype(int)
            features['age_risk_prime'] = ((df['AGE'] >= 25) & (df['AGE'] <= 45)).astype(int)
            features['age_risk_senior'] = (df['AGE'] > 45).astype(int)
        
        # 3. Vehicle Age Risk
        if 'YOM' in df.columns:
            current_year = 2025
            vehicle_age = current_year - df['YOM']
            features['vehicle_age'] = vehicle_age
            features['vehicle_old'] = (vehicle_age > 10).astype(int)
            features['vehicle_depreciation_risk'] = vehicle_age / 20  # Normalized
        
        # 4. Interest Rate Risk
        if 'EFFECTIVE_RATE' in df.columns:
            features['high_interest_rate'] = (df['EFFECTIVE_RATE'] > 15).astype(int)
            features['interest_burden'] = df['EFFECTIVE_RATE'] / 100
        
        # 5. Financial Stress Indicator
        if 'EXPENSE' in df.columns and 'INCOME' in df.columns:
            features['financial_stress'] = (df['EXPENSE'] / (df['INCOME'] + 1) > 0.8).astype(int)
        
        return pd.DataFrame(features)
    
    def create_interaction_features(self, df):
        """Create interaction features between important variables"""
        features = {}
        
        # 1. Income × Payment Rate
        if 'INCOME' in df.columns and 'PAID_RENTALS' in df.columns and 'NO_OF_RENTAL' in df.columns:
            payment_rate = df['PAID_RENTALS'] / (df['NO_OF_RENTAL'] + 1)
            features['income_payment_interaction'] = df['INCOME'] * payment_rate
        
        # 2. Age × CB Arrears
        if 'AGE' in df.columns and 'CB_ARREARS_AGE' in df.columns:
            features['age_arrears_interaction'] = df['AGE'] * df['CB_ARREARS_AGE']
        
        # 3. Interest Rate × Loan Amount
        if 'EFFECTIVE_RATE' in df.columns and 'FINANCE_AMOUNT' in df.columns:
            features['rate_amount_interaction'] = df['EFFECTIVE_RATE'] * df['FINANCE_AMOUNT']
        
        # 4. Income × Vehicle Age
        if 'INCOME' in df.columns and 'YOM' in df.columns:
            vehicle_age = 2025 - df['YOM']
            features['income_vehicle_age'] = df['INCOME'] * vehicle_age
        
        return pd.DataFrame(features)
    
    def create_statistical_features(self, df):
        """Create statistical aggregations"""
        features = {}
        
        # Polynomial features for key numeric columns
        numeric_cols = ['INCOME', 'FINANCE_AMOUNT', 'AGE']
        
        for col in numeric_cols:
            if col in df.columns:
                features[f'{col}_squared'] = df[col] ** 2
                features[f'{col}_log'] = np.log1p(df[col])  # log(1 + x) to handle zeros
        
        return pd.DataFrame(features)
    
    def fit_transform(self, X, feature_names=None):
        """
        Fit and transform training data
        X: numpy array or DataFrame
        feature_names: list of original feature names
        """
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            if feature_names is None:
                raise ValueError("feature_names required when X is numpy array")
            df = pd.DataFrame(X, columns=feature_names)
        else:
            df = X.copy()
        
        self.original_features = list(df.columns)
        
        # Create all feature groups
        financial_features = self.create_financial_ratios(df)
        payment_features = self.create_payment_features(df)
        risk_features = self.create_risk_indicators(df)
        interaction_features = self.create_interaction_features(df)
        statistical_features = self.create_statistical_features(df)
        
        # Combine all features
        all_features = pd.concat([
            df,
            financial_features,
            payment_features,
            risk_features,
            interaction_features,
            statistical_features
        ], axis=1)
        
        # Handle any NaN or inf values
        all_features = all_features.replace([np.inf, -np.inf], np.nan)
        all_features = all_features.fillna(0)
        
        # Store feature names
        self.feature_names = list(all_features.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(all_features)
        
        print(f"✅ Feature Engineering Complete:")
        print(f"   Original features: {len(self.original_features)}")
        print(f"   Engineered features: {len(self.feature_names) - len(self.original_features)}")
        print(f"   Total features: {len(self.feature_names)}")
        
        return X_scaled
    
    def transform(self, X, feature_names=None):
        """Transform new data using fitted parameters"""
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            if feature_names is None:
                feature_names = self.original_features
            df = pd.DataFrame(X, columns=feature_names)
        else:
            df = X.copy()
        
        # Create all feature groups (same as fit_transform)
        financial_features = self.create_financial_ratios(df)
        payment_features = self.create_payment_features(df)
        risk_features = self.create_risk_indicators(df)
        interaction_features = self.create_interaction_features(df)
        statistical_features = self.create_statistical_features(df)
        
        # Combine all features
        all_features = pd.concat([
            df,
            financial_features,
            payment_features,
            risk_features,
            interaction_features,
            statistical_features
        ], axis=1)
        
        # Handle any NaN or inf values
        all_features = all_features.replace([np.inf, -np.inf], np.nan)
        all_features = all_features.fillna(0)
        
        # Ensure same columns as training
        for col in self.feature_names:
            if col not in all_features.columns:
                all_features[col] = 0
        
        all_features = all_features[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(all_features)
        
        return X_scaled
    
    def save(self, filepath):
        """Save feature engineer"""
        save_dict = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'original_features': self.original_features
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"💾 Feature engineer saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load feature engineer"""
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        engineer = cls()
        engineer.scaler = save_dict['scaler']
        engineer.feature_names = save_dict['feature_names']
        engineer.original_features = save_dict['original_features']
        
        print(f"✅ Feature engineer loaded from: {filepath}")
        return engineer
    
    def get_feature_info(self):
        """Get information about created features"""
        return {
            'original_features': self.original_features,
            'total_features': len(self.feature_names),
            'engineered_features': len(self.feature_names) - len(self.original_features),
            'feature_names': self.feature_names
        }


def train_with_feature_engineering():
    """Train models with enhanced features"""
    from final_improved_models import create_model, find_optimal_threshold, evaluate_model
    
    print("="*70)
    print("🚀 TRAINING WITH FEATURE ENGINEERING")
    print("="*70)
    
    # Load data
    data_dir = Path(__file__).parent.parent / 'data' / 'processed'
    metadata_path = data_dir / 'metadata.json'
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    feature_names = metadata['feature_names']
    
    X_train = np.load(data_dir / 'X_train.npy', allow_pickle=True)
    X_val = np.load(data_dir / 'X_val.npy', allow_pickle=True)
    X_test = np.load(data_dir / 'X_test.npy', allow_pickle=True)
    y_train = np.load(data_dir / 'y_train.npy', allow_pickle=True)
    y_val = np.load(data_dir / 'y_val.npy', allow_pickle=True)
    y_test = np.load(data_dir / 'y_test.npy', allow_pickle=True)
    
    # Convert to proper format
    X_train = np.array(X_train).astype(np.float32)
    X_val = np.array(X_val).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)
    y_train = np.array([1 if str(y).upper() == 'YES' else 0 for y in y_train.ravel()])
    y_val = np.array([1 if str(y).upper() == 'YES' else 0 for y in y_val.ravel()])
    y_test = np.array([1 if str(y).upper() == 'YES' else 0 for y in y_test.ravel()])
    
    print(f"\n📊 Original Data:")
    print(f"   Training: {X_train.shape}")
    print(f"   Validation: {X_val.shape}")
    print(f"   Test: {X_test.shape}")
    
    # Apply feature engineering
    print(f"\n🔧 Applying Feature Engineering...")
    engineer = CreditRiskFeatureEngineer()
    
    X_train_eng = engineer.fit_transform(X_train, feature_names)
    X_val_eng = engineer.transform(X_val, feature_names)
    X_test_eng = engineer.transform(X_test, feature_names)
    
    print(f"\n📊 Engineered Data:")
    print(f"   Training: {X_train_eng.shape}")
    print(f"   Validation: {X_val_eng.shape}")
    print(f"   Test: {X_test_eng.shape}")
    
    # Save feature engineer
    output_dir = Path(__file__).parent.parent / 'models' / 'improved'
    output_dir.mkdir(parents=True, exist_ok=True)
    engineer.save(output_dir / 'feature_engineer.pkl')
    
    # Train XGBoost with engineered features
    print(f"\n{'='*70}")
    print(f"🔧 Training XGBoost with Engineered Features")
    print(f"{'='*70}")
    
    try:
        import xgboost as xgb
        
        model = create_model('xgboost')
        model.fit(X_train_eng, y_train, eval_set=[(X_val_eng, y_val)], verbose=False)
        print(f"✅ XGBoost trained!")
        
        # Find optimal threshold
        optimal_threshold = find_optimal_threshold(model, X_val_eng, y_val, target_recall=0.7)
        
        # Evaluate
        y_scores = model.predict_proba(X_test_eng)[:, 1]
        y_pred = (y_scores >= optimal_threshold).astype(int)
        
        results = evaluate_model(y_test, y_pred, y_scores, "XGBoost + Feature Engineering")
        
        # Save model
        model_info = {
            'model': model,
            'threshold': optimal_threshold,
            'feature_engineer': engineer,
            'results': results
        }
        
        with open(output_dir / 'xgboost_with_feature_engineering.pkl', 'wb') as f:
            pickle.dump(model_info, f)
        
        print(f"\n💾 Model with feature engineering saved!")
        
        # Save results
        with open(output_dir / 'feature_engineering_results.json', 'w') as f:
            json.dump({
                'results': results,
                'feature_info': engineer.get_feature_info()
            }, f, indent=2)
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    train_with_feature_engineering()
