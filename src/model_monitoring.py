"""
Model Monitoring and Explainability
SHAP values, feature importance, and performance tracking
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not available. Install with: pip install shap")


class ModelMonitor:
    """
    Monitor model performance and provide explainability
    """
    
    def __init__(self, model, model_name="XGBoost", feature_names=None):
        self.model = model
        self.model_name = model_name
        self.feature_names = feature_names
        self.explainer = None
        self.performance_history = []
    
    def initialize_explainer(self, X_background=None):
        """Initialize SHAP explainer"""
        if not SHAP_AVAILABLE:
            print("❌ SHAP not available")
            return False
        
        try:
            print("🔧 Initializing SHAP explainer...")
            
            # Use a subset of data as background
            if X_background is not None:
                if len(X_background) > 100:
                    background = X_background[:100]
                else:
                    background = X_background
            else:
                background = None
            
            # For tree-based models, use TreeExplainer
            if hasattr(self.model, 'get_booster'):  # XGBoost
                self.explainer = shap.TreeExplainer(self.model)
            elif hasattr(self.model, 'tree_'):  # sklearn tree models
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # Fallback to KernelExplainer
                if background is not None:
                    self.explainer = shap.KernelExplainer(
                        self.model.predict_proba,
                        background
                    )
            
            print("✅ SHAP explainer initialized")
            return True
            
        except Exception as e:
            print(f"⚠️ Could not initialize SHAP explainer: {e}")
            return False
    
    def get_feature_importance(self, method='gain'):
        """
        Get feature importance
        method: 'gain', 'weight', 'cover', or 'shap'
        """
        try:
            if method == 'shap' and self.explainer is not None:
                # SHAP-based importance (requires background data)
                return None  # Need to call explain_instance first
            
            # Model-based importance
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                
                importance_df = pd.DataFrame({
                    'feature': self.feature_names if self.feature_names else [f'f{i}' for i in range(len(importances))],
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                return importance_df
            
            return None
            
        except Exception as e:
            print(f"⚠️ Error getting feature importance: {e}")
            return None
    
    def plot_feature_importance(self, top_n=20, save_path=None):
        """Plot feature importance"""
        importance_df = self.get_feature_importance()
        
        if importance_df is None:
            print("❌ Could not get feature importance")
            return
        
        # Get top N features
        top_features = importance_df.head(top_n)
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance', fontweight='bold')
        plt.title(f'Top {top_n} Feature Importances - {self.model_name}', 
                  fontweight='bold', fontsize=14)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Feature importance plot saved: {save_path}")
        
        plt.show()
        
        return top_features
    
    def explain_instance(self, X, instance_idx=0):
        """
        Explain a single prediction using SHAP
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            print("❌ SHAP explainer not available")
            return None
        
        try:
            # Get SHAP values
            shap_values = self.explainer.shap_values(X)
            
            # For binary classification, get values for positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            return shap_values
            
        except Exception as e:
            print(f"⚠️ Error explaining instance: {e}")
            return None
    
    def plot_shap_summary(self, X, max_display=20, save_path=None):
        """Plot SHAP summary"""
        if not SHAP_AVAILABLE or self.explainer is None:
            print("❌ SHAP explainer not available")
            return
        
        try:
            print("🔧 Computing SHAP values...")
            
            # Limit samples for speed
            if len(X) > 500:
                X_sample = X[:500]
            else:
                X_sample = X
            
            shap_values = self.explainer.shap_values(X_sample)
            
            # For binary classification, get values for positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Create summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=self.feature_names,
                max_display=max_display,
                show=False
            )
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"💾 SHAP summary plot saved: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Error creating SHAP summary: {e}")
    
    def plot_shap_waterfall(self, X, instance_idx=0, save_path=None):
        """Plot SHAP waterfall for a single prediction"""
        if not SHAP_AVAILABLE or self.explainer is None:
            print("❌ SHAP explainer not available")
            return
        
        try:
            shap_values = self.explainer(X[instance_idx:instance_idx+1])
            
            # For binary classification
            if len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]
            
            plt.figure(figsize=(12, 8))
            shap.plots.waterfall(shap_values[0], max_display=15, show=False)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"💾 SHAP waterfall plot saved: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Error creating SHAP waterfall: {e}")
    
    def plot_shap_force(self, X, instance_idx=0, save_path=None):
        """Plot SHAP force plot for a single prediction"""
        if not SHAP_AVAILABLE or self.explainer is None:
            print("❌ SHAP explainer not available")
            return
        
        try:
            shap_values = self.explainer.shap_values(X[instance_idx:instance_idx+1])
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Get expected value
            if isinstance(self.explainer.expected_value, np.ndarray):
                expected_value = self.explainer.expected_value[1]
            else:
                expected_value = self.explainer.expected_value
            
            shap.force_plot(
                expected_value,
                shap_values[0],
                X[instance_idx],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"💾 SHAP force plot saved: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Error creating SHAP force plot: {e}")
    
    def analyze_prediction(self, X, instance_idx=0, feature_names=None):
        """
        Comprehensive analysis of a single prediction
        """
        print(f"\n{'='*70}")
        print(f"🔍 PREDICTION ANALYSIS - Instance {instance_idx}")
        print(f"{'='*70}")
        
        # Get prediction
        prob = self.model.predict_proba(X[instance_idx:instance_idx+1])[0, 1]
        pred = "DEFAULT" if prob >= 0.5 else "NO_DEFAULT"
        
        print(f"\n📊 Prediction:")
        print(f"   Class: {pred}")
        print(f"   Default Probability: {prob:.4f} ({prob*100:.2f}%)")
        print(f"   Risk Score: {prob*100:.2f}/100")
        
        # Feature importance for this instance
        importance_df = self.get_feature_importance()
        if importance_df is not None:
            print(f"\n🔝 Top 10 Most Important Features (Global):")
            for idx, row in importance_df.head(10).iterrows():
                print(f"   {idx+1}. {row['feature']}: {row['importance']:.4f}")
        
        # SHAP explanation
        if SHAP_AVAILABLE and self.explainer is not None:
            print(f"\n🎯 SHAP Analysis (Local Explanation):")
            try:
                shap_values = self.explainer.shap_values(X[instance_idx:instance_idx+1])
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1][0]
                else:
                    shap_values = shap_values[0]
                
                # Get top contributing features
                abs_shap = np.abs(shap_values)
                top_indices = np.argsort(abs_shap)[-10:][::-1]
                
                print(f"   Top 10 Features Driving This Prediction:")
                for i, idx in enumerate(top_indices, 1):
                    feature_name = self.feature_names[idx] if self.feature_names else f"Feature {idx}"
                    shap_val = shap_values[idx]
                    direction = "increases" if shap_val > 0 else "decreases"
                    print(f"   {i}. {feature_name}: {shap_val:+.4f} ({direction} default risk)")
                
            except Exception as e:
                print(f"   ⚠️ Could not compute SHAP values: {e}")
    
    def track_performance(self, y_true, y_pred, y_scores, dataset_name="test"):
        """Track performance metrics over time"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, precision_recall_curve, auc, confusion_matrix
        )
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_scores)
        
        prec_vals, rec_vals, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(rec_vals, prec_vals)
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Store performance
        performance = {
            'timestamp': datetime.now().isoformat(),
            'dataset': dataset_name,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'confusion_matrix': cm.tolist(),
            'n_samples': len(y_true),
            'n_positives': int(y_true.sum()),
            'n_negatives': int(len(y_true) - y_true.sum())
        }
        
        self.performance_history.append(performance)
        
        return performance
    
    def plot_performance_over_time(self, save_path=None):
        """Plot performance metrics over time"""
        if not self.performance_history:
            print("❌ No performance history available")
            return
        
        df = pd.DataFrame(self.performance_history)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Recall over time
        axes[0, 0].plot(df.index, df['recall'], marker='o', linewidth=2, color='steelblue')
        axes[0, 0].set_ylabel('Recall', fontweight='bold')
        axes[0, 0].set_title('Recall Over Time', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0.7, color='g', linestyle='--', label='Target (70%)')
        axes[0, 0].legend()
        
        # Precision over time
        axes[0, 1].plot(df.index, df['precision'], marker='o', linewidth=2, color='coral')
        axes[0, 1].set_ylabel('Precision', fontweight='bold')
        axes[0, 1].set_title('Precision Over Time', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1-Score over time
        axes[1, 0].plot(df.index, df['f1_score'], marker='o', linewidth=2, color='mediumseagreen')
        axes[1, 0].set_ylabel('F1-Score', fontweight='bold')
        axes[1, 0].set_title('F1-Score Over Time', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # ROC AUC over time
        axes[1, 1].plot(df.index, df['roc_auc'], marker='o', linewidth=2, color='mediumpurple')
        axes[1, 1].set_ylabel('ROC AUC', fontweight='bold')
        axes[1, 1].set_title('ROC AUC Over Time', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0.85, color='g', linestyle='--', label='Good (0.85)')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Performance over time plot saved: {save_path}")
        
        plt.show()
    
    def generate_monitoring_report(self, save_path=None):
        """Generate comprehensive monitoring report"""
        report = {
            'model_name': self.model_name,
            'report_timestamp': datetime.now().isoformat(),
            'performance_history': self.performance_history,
            'summary': {}
        }
        
        if self.performance_history:
            latest = self.performance_history[-1]
            report['summary'] = {
                'latest_performance': latest,
                'total_evaluations': len(self.performance_history)
            }
        
        # Get feature importance
        importance_df = self.get_feature_importance()
        if importance_df is not None:
            report['feature_importance'] = importance_df.to_dict('records')
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"💾 Monitoring report saved: {save_path}")
        
        return report
    
    def save_state(self, filepath):
        """Save monitor state"""
        state = {
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'performance_history': self.performance_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"💾 Monitor state saved: {filepath}")
    
    @classmethod
    def load_state(cls, filepath, model):
        """Load monitor state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        monitor = cls(
            model=model,
            model_name=state['model_name'],
            feature_names=state['feature_names']
        )
        monitor.performance_history = state['performance_history']
        
        print(f"✅ Monitor state loaded: {filepath}")
        return monitor


def create_monitoring_dashboard():
    """Create comprehensive monitoring dashboard"""
    print("="*70)
    print("🔍 MODEL MONITORING & EXPLAINABILITY")
    print("="*70)
    
    # Load model
    model_path = Path(__file__).parent.parent / 'models' / 'improved' / 'rank1_xgboost.pkl'
    
    with open(model_path, 'rb') as f:
        model_info = pickle.load(f)
    
    model = model_info['model']
    threshold = model_info['threshold']
    
    # Load data
    data_dir = Path(__file__).parent.parent / 'data' / 'processed'
    
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    feature_names = metadata['feature_names']
    
    X_test = np.load(data_dir / 'X_test.npy', allow_pickle=True)
    y_test = np.load(data_dir / 'y_test.npy', allow_pickle=True)
    
    X_test = np.array(X_test).astype(np.float32)
    y_test = np.array([1 if str(y).upper() == 'YES' else 0 for y in y_test.ravel()])
    
    print(f"\n✅ Model and data loaded")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {len(feature_names)}")
    
    # Create monitor
    monitor = ModelMonitor(model, "XGBoost", feature_names)
    
    # Initialize SHAP explainer
    monitor.initialize_explainer(X_test)
    
    # Output directory
    output_dir = Path(__file__).parent.parent / 'results' / 'monitoring'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Feature Importance
    print(f"\n{'='*70}")
    print(f"1️⃣ Feature Importance Analysis")
    print(f"{'='*70}")
    
    top_features = monitor.plot_feature_importance(
        top_n=20,
        save_path=output_dir / 'feature_importance.png'
    )
    
    # 2. SHAP Summary
    if SHAP_AVAILABLE:
        print(f"\n{'='*70}")
        print(f"2️⃣ SHAP Summary Plot")
        print(f"{'='*70}")
        
        monitor.plot_shap_summary(
            X_test,
            max_display=20,
            save_path=output_dir / 'shap_summary.png'
        )
    
    # 3. Individual Prediction Analysis
    print(f"\n{'='*70}")
    print(f"3️⃣ Individual Prediction Analysis")
    print(f"{'='*70}")
    
    # Find a default case
    default_indices = np.where(y_test == 1)[0]
    if len(default_indices) > 0:
        default_idx = default_indices[0]
        monitor.analyze_prediction(X_test, default_idx, feature_names)
        
        if SHAP_AVAILABLE:
            monitor.plot_shap_waterfall(
                X_test,
                default_idx,
                save_path=output_dir / 'shap_waterfall_default.png'
            )
    
    # Find a non-default case
    non_default_indices = np.where(y_test == 0)[0]
    if len(non_default_indices) > 0:
        non_default_idx = non_default_indices[0]
        monitor.analyze_prediction(X_test, non_default_idx, feature_names)
        
        if SHAP_AVAILABLE:
            monitor.plot_shap_waterfall(
                X_test,
                non_default_idx,
                save_path=output_dir / 'shap_waterfall_non_default.png'
            )
    
    # 4. Track Performance
    print(f"\n{'='*70}")
    print(f"4️⃣ Performance Tracking")
    print(f"{'='*70}")
    
    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= threshold).astype(int)
    
    performance = monitor.track_performance(y_test, y_pred, y_scores, "test_set")
    
    print(f"\n📊 Current Performance:")
    print(f"   Recall: {performance['recall']:.4f}")
    print(f"   Precision: {performance['precision']:.4f}")
    print(f"   F1-Score: {performance['f1_score']:.4f}")
    print(f"   ROC AUC: {performance['roc_auc']:.4f}")
    print(f"   PR AUC: {performance['pr_auc']:.4f}")
    
    # 5. Generate Report
    print(f"\n{'='*70}")
    print(f"5️⃣ Generating Monitoring Report")
    print(f"{'='*70}")
    
    report = monitor.generate_monitoring_report(
        save_path=output_dir / 'monitoring_report.json'
    )
    
    # Save monitor state
    monitor.save_state(output_dir / 'monitor_state.pkl')
    
    print(f"\n✅ Monitoring dashboard complete!")
    print(f"📁 Results saved to: {output_dir}")
    
    return monitor


if __name__ == "__main__":
    # Install SHAP if not available
    if not SHAP_AVAILABLE:
        print("\n⚠️ SHAP not installed. Installing now...")
        import subprocess
        subprocess.run(["pip", "install", "shap"])
        print("✅ SHAP installed. Please restart the script.")
    else:
        create_monitoring_dashboard()
