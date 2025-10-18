
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from autoencoder_training import CreditRiskAutoencoder, AutoencoderTrainer
from data_preprocessing import CreditRiskPreprocessor


class CreditRiskScorer:
    """
    Convert autoencoder reconstruction errors to actionable risk scores
    """
    
    # Risk thresholds (customize these for your business)
    RISK_THRESHOLDS = {
        'LOW': (0, 30),       # 0-30: Low risk → Approve
        'MEDIUM': (30, 60),   # 30-60: Medium risk → Approve with conditions
        'HIGH': (60, 100)     # 60-100: High risk → Reject
    }
    
    def __init__(
        self,
        model_path: str = None,
        preprocessor_path: str = None,
        device: str = None
    ):
        """
        Initialize risk scorer
        
        Args:
            model_path: Path to trained autoencoder (.pth)
            preprocessor_path: Path to fitted preprocessor (.pkl)
            device: 'cuda', 'cpu', or None (auto)
        """
        self.device = torch.device(
            device if device else 
            ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        self.model = None
        self.preprocessor = None
        self.threshold = None
        self.error_min = None
        self.error_max = None
        
        # Load model and preprocessor if provided
        if model_path:
            self.load_model(model_path)
        if preprocessor_path:
            self.load_preprocessor(preprocessor_path)
    
    def load_model(self, model_path: str):
        """Load trained autoencoder model"""
        print(f"📂 Loading model from: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Get model architecture
        input_dim = checkpoint['input_dim']
        encoding_dims = checkpoint['encoding_dims']
        
        # Initialize and load model
        self.model = CreditRiskAutoencoder(input_dim, encoding_dims).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load threshold
        self.threshold = checkpoint.get('threshold')
        
        print(f"✅ Model loaded successfully")
        print(f"   Input dim: {input_dim}")
        print(f"   Architecture: {encoding_dims}")
        if self.threshold:
            print(f"   Threshold: {self.threshold:.6f}")
    
    def load_preprocessor(self, preprocessor_path: str):
        """Load fitted preprocessor"""
        print(f"📂 Loading preprocessor from: {preprocessor_path}")
        
        self.preprocessor = CreditRiskPreprocessor.load(preprocessor_path)
        
        print(f"✅ Preprocessor loaded successfully")
    
    def calibrate_error_range(
        self,
        errors_normal: np.ndarray,
        errors_default: np.ndarray = None,
        percentile_min: float = 1,
        percentile_max: float = 99
    ):
        """
        Calibrate error range for 0-100 scaling
        
        Args:
            errors_normal: Reconstruction errors from normal cases
            errors_default: Reconstruction errors from default cases (optional)
            percentile_min: Lower percentile for scaling (default: 1%)
            percentile_max: Upper percentile for scaling (default: 99%)
        """
        print(f"\n{'='*70}")
        print("CALIBRATING RISK SCORE RANGE")
        print(f"{'='*70}")
        
        # Combine errors if defaults provided
        if errors_default is not None:
            all_errors = np.concatenate([errors_normal, errors_default])
        else:
            all_errors = errors_normal
        
        # Set min/max based on percentiles (to avoid outliers)
        self.error_min = np.percentile(all_errors, percentile_min)
        self.error_max = np.percentile(all_errors, percentile_max)
        
        print(f"📊 Error Statistics:")
        print(f"   Min (1st percentile): {self.error_min:.6f}")
        print(f"   Max (99th percentile): {self.error_max:.6f}")
        print(f"   Mean: {np.mean(all_errors):.6f}")
        print(f"   Std: {np.std(all_errors):.6f}")
        
        print(f"\n✅ Calibration complete!")
        print(f"   Errors will be mapped to 0-100 risk score")
        print(f"{'='*70}\n")
    
    def error_to_risk_score(self, error: float) -> float:
        """
        Convert reconstruction error to risk score (0-100)
        
        Args:
            error: Reconstruction error
            
        Returns:
            Risk score (0-100)
        """
        if self.error_min is None or self.error_max is None:
            raise ValueError(
                "Error range not calibrated! Run calibrate_error_range() first."
            )
        
        # Clip to calibrated range
        error = np.clip(error, self.error_min, self.error_max)
        
        # Scale to 0-100
        risk_score = 100 * (error - self.error_min) / (self.error_max - self.error_min)
        
        return float(risk_score)
    
    def categorize_risk(self, risk_score: float) -> Tuple[str, str]:
        """
        Categorize risk score
        
        Args:
            risk_score: Risk score (0-100)
            
        Returns:
            (category, action)
        """
        if risk_score < self.RISK_THRESHOLDS['LOW'][1]:
            return 'LOW', 'APPROVE'
        elif risk_score < self.RISK_THRESHOLDS['MEDIUM'][1]:
            return 'MEDIUM', 'APPROVE_WITH_CONDITIONS'
        else:
            return 'HIGH', 'REJECT'
    
    def calculate_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction error for samples
        
        Args:
            X: Preprocessed features (numpy array)
            
        Returns:
            Reconstruction errors
        """
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            reconstructed = self.model(X_tensor)
            
            # MSE per sample
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            
        return errors.cpu().numpy()
    
    def predict(
        self, 
        customer_data: pd.DataFrame,
        return_details: bool = True
    ) -> Dict:
        """
        Predict risk for customer application
        
        Args:
            customer_data: DataFrame with customer features
            return_details: Include detailed breakdown
            
        Returns:
            Dictionary with risk assessment
        """
        if self.model is None:
            raise ValueError("Model not loaded! Call load_model() first.")
        
        if self.preprocessor is None:
            raise ValueError("Preprocessor not loaded! Call load_preprocessor() first.")
        
        # Preprocess data
        X = self.preprocessor.transform_new_data(customer_data)
        
        # Calculate reconstruction error
        errors = self.calculate_reconstruction_error(X)
        
        # Convert to risk scores
        risk_scores = [self.error_to_risk_score(e) for e in errors]
        
        # Categorize risks
        results = []
        for i, risk_score in enumerate(risk_scores):
            category, action = self.categorize_risk(risk_score)
            
            result = {
                'risk_score': round(risk_score, 2),
                'risk_category': category,
                'action': action,
                'reconstruction_error': float(errors[i])
            }
            
            if return_details:
                result['details'] = {
                    'threshold': self.threshold,
                    'exceeds_threshold': errors[i] > self.threshold,
                    'confidence': self._calculate_confidence(risk_score)
                }
            
            results.append(result)
        
        return results[0] if len(results) == 1 else results
    
    def predict_batch(
        self,
        customer_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Predict risk for multiple customers
        
        Args:
            customer_data: DataFrame with customer features
            
        Returns:
            DataFrame with risk assessments
        """
        results = self.predict(customer_data, return_details=False)
        
        if not isinstance(results, list):
            results = [results]
        
        # Create results DataFrame
        df_results = pd.DataFrame(results)
        
        # Combine with original data
        df_combined = pd.concat([
            customer_data.reset_index(drop=True),
            df_results
        ], axis=1)
        
        return df_combined
    
    def _calculate_confidence(self, risk_score: float) -> str:
        """Calculate confidence level based on risk score"""
        # Distance from category boundaries
        if risk_score < 15 or risk_score > 85:
            return "HIGH"  # Far from boundaries
        elif risk_score < 25 or risk_score > 75:
            return "MEDIUM"
        else:
            return "LOW"  # Near boundaries (30, 60)
    
    def print_risk_assessment(self, result: Dict):
        """Pretty print risk assessment"""
        print(f"\n{'='*60}")
        print("CREDIT RISK ASSESSMENT")
        print(f"{'='*60}")
        
        # Risk score with visual bar
        score = result['risk_score']
        bar_length = int(score / 2)
        bar = '█' * bar_length + '░' * (50 - bar_length)
        
        print(f"\n🎯 RISK SCORE: {score:.1f}/100")
        print(f"   [{bar}]")
        
        # Category
        category = result['risk_category']
        category_emoji = {
            'LOW': '🟢',
            'MEDIUM': '🟡',
            'HIGH': '🔴'
        }
        
        print(f"\n📊 RISK CATEGORY: {category_emoji[category]} {category}")
        
        # Action
        action = result['action']
        action_emoji = {
            'APPROVE': '✅',
            'APPROVE_WITH_CONDITIONS': '⚠️',
            'REJECT': '❌'
        }
        
        print(f"\n💼 RECOMMENDED ACTION: {action_emoji[action]} {action}")
        
        # Thresholds
        print(f"\n📏 RISK THRESHOLDS:")
        for cat, (low, high) in self.RISK_THRESHOLDS.items():
            indicator = "◄ YOU ARE HERE" if cat == category else ""
            print(f"   • {cat:6s}: {low:2d}-{high:3d} {indicator}")
        
        # Details if available
        if 'details' in result:
            details = result['details']
            print(f"\n🔍 DETAILS:")
            print(f"   • Reconstruction Error: {result['reconstruction_error']:.6f}")
            print(f"   • Threshold: {details['threshold']:.6f}")
            print(f"   • Exceeds Threshold: {details['exceeds_threshold']}")
            print(f"   • Confidence: {details['confidence']}")
        
        print(f"\n{'='*60}\n")


def create_risk_report(
    results: List[Dict],
    save_path: str = None
) -> pd.DataFrame:
    """
    Create comprehensive risk report
    
    Args:
        results: List of prediction results
        save_path: Path to save CSV report
        
    Returns:
        DataFrame with full report
    """
    df = pd.DataFrame(results)
    
    # Add summary statistics
    print("\n" + "="*70)
    print("RISK ASSESSMENT SUMMARY")
    print("="*70)
    
    print("\n📊 Risk Distribution:")
    category_counts = df['risk_category'].value_counts()
    for cat in ['LOW', 'MEDIUM', 'HIGH']:
        count = category_counts.get(cat, 0)
        pct = count / len(df) * 100
        print(f"   {cat:6s}: {count:4d} ({pct:5.1f}%)")
    
    print("\n💼 Action Distribution:")
    action_counts = df['action'].value_counts()
    for action in ['APPROVE', 'APPROVE_WITH_CONDITIONS', 'REJECT']:
        count = action_counts.get(action, 0)
        pct = count / len(df) * 100
        print(f"   {action:23s}: {count:4d} ({pct:5.1f}%)")
    
    print("\n📈 Risk Score Statistics:")
    print(f"   Mean:   {df['risk_score'].mean():.2f}")
    print(f"   Median: {df['risk_score'].median():.2f}")
    print(f"   Std:    {df['risk_score'].std():.2f}")
    print(f"   Min:    {df['risk_score'].min():.2f}")
    print(f"   Max:    {df['risk_score'].max():.2f}")
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\n💾 Report saved to: {save_path}")
    
    print("="*70 + "\n")
    
    return df


# ============================================================
# EXAMPLE USAGE
# ============================================================

def example_single_prediction():
    """Example: Single customer prediction"""
    print("="*70)
    print("EXAMPLE 1: SINGLE CUSTOMER PREDICTION")
    print("="*70)
    
    # Initialize scorer
    scorer = CreditRiskScorer(
        model_path='models/autoencoder/default_autoencoder.pth',
        preprocessor_path='models/preprocessor/preprocessor.pkl'
    )
    
    # Calibrate error range (run once, ideally on validation set)
    # Load validation errors from training
    val_errors = np.load('results/validation_errors.npy')  # Save during training
    scorer.calibrate_error_range(val_errors)
    
    # New customer application
    customer = pd.DataFrame({
        'age': [35],
        'annual_income': [55000],
        'debt': [18000],
        'loan_amount': [250000],
        'credit_score': [680],
        'employment_length': [5],
        'employment_type': ['Full-time']
    })
    
    # Predict
    result = scorer.predict(customer)
    
    # Display
    scorer.print_risk_assessment(result)
    
    return result


def example_batch_prediction():
    """Example: Batch prediction for multiple customers"""
    print("="*70)
    print("EXAMPLE 2: BATCH PREDICTION")
    print("="*70)
    
    # Initialize scorer
    scorer = CreditRiskScorer(
        model_path='models/autoencoder/lending_club_autoencoder.pth',
        preprocessor_path='models/preprocessor/lending_club/preprocessor.pkl'
    )
    
    # Calibrate
    val_errors = np.load('results/validation_errors.npy')
    scorer.calibrate_error_range(val_errors)
    
    # Multiple customers
    customers = pd.DataFrame({
        'age': [35, 45, 28, 52],
        'annual_income': [55000, 85000, 35000, 95000],
        'debt': [18000, 15000, 22000, 12000],
        'loan_amount': [250000, 400000, 180000, 350000],
        'credit_score': [680, 750, 610, 780],
        'employment_length': [5, 12, 2, 15],
        'employment_type': ['Full-time', 'Self-employed', 'Part-time', 'Full-time']
    })
    
    # Predict batch
    results_df = scorer.predict_batch(customers)
    
    print("\n📊 BATCH RESULTS:")
    print(results_df[['age', 'credit_score', 'risk_score', 'risk_category', 'action']])
    
    # Create report
    results_list = results_df[['risk_score', 'risk_category', 'action', 'reconstruction_error']].to_dict('records')
    report = create_risk_report(results_list, 'results/risk_report.csv')
    
    return results_df


if __name__ == "__main__":
    """
    Run examples
    """
    print("""                                                                 
    CREDIT RISK SCORING SYSTEM                             
    Autoencoder-based Risk Assessment                       
                                                                      
    Risk Categories:                                                
        🟢 LOW (0-30):      Approve                                  
        🟡 MEDIUM (30-60):  Approve with conditions                  
        🔴 HIGH (60-100):   Reject                                   
    """)
    
    # Run examples
    try:
        example_single_prediction()
    except Exception as e:
        print(f"⚠️ Example 1 failed: {e}")
        print("Make sure model and preprocessor are trained and saved!")
    
    # Uncomment to run batch example
    # example_batch_prediction()