"""
Final Optimized Credit Risk Models
Addresses autoencoder's poor performance with supervised ML
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc, f1_score, recall_score, precision_score,
    roc_curve
)
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available")


def create_model(model_type):
    """Create model with optimized hyperparameters for imbalanced data"""
    
    if model_type == 'xgboost' and XGBOOST_AVAILABLE:
        # XGBoost with scale_pos_weight handles class imbalance
        # scale_pos_weight = (negative samples) / (positive samples) = 68827 / 701 ≈ 98
        return xgb.XGBClassifier(
            scale_pos_weight=1.0,  # Already balanced in training data
            max_depth=6,
            learning_rate=0.05,
            n_estimators=200,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='aucpr',
            random_state=42,
            tree_method='hist',
            n_jobs=-1
        )
        
    elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
        return lgb.LGBMClassifier(
            scale_pos_weight=1.0,
            max_depth=6,
            learning_rate=0.05,
            n_estimators=200,
            min_child_weight=0.001,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary',
            metric='auc',
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        
    elif model_type == 'random_forest':
        return RandomForestClassifier(
            n_estimators=100,  # Reduced for speed
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight=None,  # Already balanced
            random_state=42,
            n_jobs=-1,
            max_features='sqrt'
        )
        
    elif model_type == 'gradient_boosting':
        return GradientBoostingClassifier(
            n_estimators=100,  # Reduced for speed
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42,
            max_features='sqrt'
        )
        
    elif model_type == 'logistic':
        return LogisticRegression(
            class_weight=None,  # Already balanced
            max_iter=1000,
            C=0.1,
            random_state=42,
            n_jobs=-1
        )
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def find_optimal_threshold(model, X_val, y_val, target_recall=0.7):
    """Find threshold that achieves target recall"""
    probs = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
    
    # Find threshold closest to target recall
    valid_indices = recalls >= target_recall
    if not any(valid_indices):
        # If can't achieve target, find best available
        idx = 0
    else:
        # Among thresholds achieving target recall, pick one with best precision
        valid_precs = precisions[valid_indices]
        valid_thresh_indices = np.where(valid_indices)[0]
        best_valid_idx = valid_thresh_indices[np.argmax(valid_precs)]
        idx = best_valid_idx
    
    optimal_threshold = thresholds[idx] if idx < len(thresholds) else 0.5
    
    print(f"\n🎯 Optimal threshold for {target_recall*100:.0f}% recall: {optimal_threshold:.4f}")
    print(f"   Expected Recall: {recalls[idx]:.4f}, Precision: {precisions[idx]:.4f}")
    
    return optimal_threshold


def evaluate_model(y_true, y_pred, y_scores, model_name="Model"):
    """Comprehensive evaluation"""
    print(f"\n{'='*70}")
    print(f"📊 {model_name} Evaluation Results")
    print(f"{'='*70}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Advanced metrics
    roc_auc = roc_auc_score(y_true, y_scores)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall_vals, precision_vals)
    
    # Display results
    print(f"\n📋 Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                 No    Yes")
    print(f"   Actual  No  {tn:5d}  {fp:5d}")
    print(f"          Yes  {fn:5d}  {tp:5d}")
    
    print(f"\n📈 Classification Metrics:")
    print(f"  ✓ Accuracy:  {accuracy:.4f}")
    print(f"  ✓ Precision: {precision:.4f} (When we predict default, how often correct?)")
    print(f"  ⭐ Recall:   {recall:.4f} (Of all actual defaults, how many caught?) {'✅' if recall >= 0.7 else '⚠️' if recall >= 0.5 else '❌'}")
    print(f"  ✓ F1-Score:  {f1:.4f}")
    
    print(f"\n💰 Business Impact (on {len(y_true)} loans):")
    total_defaults = tp + fn
    print(f"  ✅ Defaults Caught:  {tp:4d} / {total_defaults:4d} ({recall*100:5.1f}%)")
    print(f"  ❌ Defaults Missed:  {fn:4d} / {total_defaults:4d} ({(fn/total_defaults)*100:5.1f}%)")
    print(f"  ⚠️  False Alarms:    {fp:4d} (Good customers incorrectly flagged)")
    
    print(f"\n📊 Advanced Metrics:")
    print(f"  ROC AUC:        {roc_auc:.4f} {'✅' if roc_auc > 0.85 else '⚠️' if roc_auc > 0.7 else '❌'}")
    print(f"  PR AUC:         {pr_auc:.4f} {'✅' if pr_auc > 0.3 else '⚠️' if pr_auc > 0.1 else '❌'}")
    
    # Assessment
    print(f"\n💡 Overall Assessment:")
    if recall >= 0.7 and pr_auc > 0.3:
        print(f"  ✅ EXCELLENT: Catches {recall*100:.0f}% of defaults - Production Ready!")
    elif recall >= 0.6:
        print(f"  ✅ GOOD: Catches {recall*100:.0f}% of defaults - Usable for production")
    elif recall >= 0.5:
        print(f"  ⚠️  MODERATE: Catches {recall*100:.0f}% of defaults - Could be improved")
    else:
        print(f"  ❌ POOR: Only catches {recall*100:.0f}% of defaults - Not production ready")
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_negatives': int(tn),
        'confusion_matrix': cm.tolist()
    }


def plot_comparison(results_dict, y_test, save_dir):
    """Create visualization comparing all models"""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    models = list(results_dict.keys())
    
    # 1. Recall Comparison (MOST IMPORTANT)
    ax1 = fig.add_subplot(gs[0, 0])
    recalls = [results_dict[m]['recall'] for m in models]
    colors = ['green' if r >= 0.7 else 'orange' if r >= 0.5 else 'red' for r in recalls]
    bars = ax1.barh(models, recalls, color=colors)
    ax1.set_xlabel('Recall (% Defaults Caught)', fontweight='bold', fontsize=11)
    ax1.set_title('⭐ Recall - Most Important Metric', fontweight='bold', fontsize=12)
    ax1.axvline(x=0.7, color='green', linestyle='--', alpha=0.5, label='Excellent (70%)')
    ax1.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate (50%)')
    ax1.set_xlim([0, 1])
    ax1.legend()
    for i, (bar, val) in enumerate(zip(bars, recalls)):
        ax1.text(val + 0.02, i, f'{val:.1%}', va='center', fontweight='bold')
    
    # 2. PR AUC Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    pr_aucs = [results_dict[m]['pr_auc'] for m in models]
    colors = ['green' if p >= 0.3 else 'orange' if p >= 0.1 else 'red' for p in pr_aucs]
    bars = ax2.barh(models, pr_aucs, color=colors)
    ax2.set_xlabel('Precision-Recall AUC', fontweight='bold', fontsize=11)
    ax2.set_title('PR AUC (For Imbalanced Data)', fontweight='bold', fontsize=12)
    ax2.axvline(x=0.3, color='green', linestyle='--', alpha=0.5, label='Good (0.3)')
    ax2.set_xlim([0, 1])
    ax2.legend()
    for i, (bar, val) in enumerate(zip(bars, pr_aucs)):
        ax2.text(val + 0.02, i, f'{val:.3f}', va='center', fontweight='bold')
    
    # 3. F1-Score Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    f1_scores = [results_dict[m]['f1_score'] for m in models]
    ax3.barh(models, f1_scores, color='steelblue')
    ax3.set_xlabel('F1-Score', fontweight='bold', fontsize=11)
    ax3.set_title('F1-Score Comparison', fontweight='bold', fontsize=12)
    ax3.set_xlim([0, 1])
    for i, val in enumerate(f1_scores):
        ax3.text(val + 0.02, i, f'{val:.3f}', va='center')
    
    # 4. Defaults Caught vs Missed
    ax4 = fig.add_subplot(gs[1, 0])
    tps = [results_dict[m]['true_positives'] for m in models]
    total_defaults = y_test.sum()
    x = np.arange(len(models))
    width = 0.35
    ax4.barh(x, tps, width, label='Caught', color='green', alpha=0.7)
    ax4.barh(x, [total_defaults - tp for tp in tps], width, left=tps, label='Missed', color='red', alpha=0.7)
    ax4.set_yticks(x)
    ax4.set_yticklabels(models)
    ax4.set_xlabel('Number of Defaults', fontweight='bold', fontsize=11)
    ax4.set_title(f'Defaults: Caught vs Missed (Total: {total_defaults})', fontweight='bold', fontsize=12)
    ax4.legend()
    
    # 5. ROC AUC Comparison
    ax5 = fig.add_subplot(gs[1, 1])
    roc_aucs = [results_dict[m]['roc_auc'] for m in models]
    ax5.barh(models, roc_aucs, color='coral')
    ax5.set_xlabel('ROC AUC', fontweight='bold', fontsize=11)
    ax5.set_title('ROC AUC Comparison', fontweight='bold', fontsize=12)
    ax5.axvline(x=0.85, color='green', linestyle='--', alpha=0.5, label='Good (0.85)')
    ax5.set_xlim([0, 1])
    ax5.legend()
    
    # 6. Precision vs Recall Trade-off
    ax6 = fig.add_subplot(gs[1, 2])
    precisions = [results_dict[m]['precision'] for m in models]
    ax6.scatter(recalls, precisions, s=300, alpha=0.6, c=range(len(models)), cmap='viridis')
    for i, model in enumerate(models):
        ax6.annotate(model, (recalls[i], precisions[i]), fontsize=9, ha='center', fontweight='bold')
    ax6.set_xlabel('Recall (Defaults Caught)', fontweight='bold', fontsize=11)
    ax6.set_ylabel('Precision (Accuracy of Predictions)', fontweight='bold', fontsize=11)
    ax6.set_title('Precision vs Recall Trade-off', fontweight='bold', fontsize=12)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim([0, 1])
    ax6.set_ylim([0, 1])
    
    plt.suptitle('🏆 Credit Risk Model Comparison\nCompared to Autoencoder (2.5% recall), these models are vastly superior', 
                 fontsize=14, fontweight='bold')
    
    save_path = save_dir / 'model_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved: {save_path}")
    plt.close()


def main():
    """Main training pipeline"""
    
    print("="*70)
    print("🚀 IMPROVED CREDIT RISK MODEL TRAINING")
    print("   Supervised ML models to replace failing autoencoder")
    print("="*70)
    
    # Load data
    data_dir = Path(__file__).parent.parent / 'data' / 'processed'
    
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
    
    print(f"\n✅ Data Loaded:")
    print(f"  Training:   {X_train.shape[0]:6d} samples ({y_train.sum():5d} defaults, {y_train.mean()*100:.2f}%)")
    print(f"  Validation: {X_val.shape[0]:6d} samples ({y_val.sum():5d} defaults, {y_val.mean()*100:.2f}%)")
    print(f"  Test:       {X_test.shape[0]:6d} samples ({y_test.sum():5d} defaults, {y_test.mean()*100:.2f}%)")
    print(f"\n📝 Note: Training data is balanced (SMOTE applied), test data is imbalanced (real-world)")
    
    # Models to train
    model_types = []
    if XGBOOST_AVAILABLE:
        model_types.append(('xgboost', 'XGBoost'))
    if LIGHTGBM_AVAILABLE:
        model_types.append(('lightgbm', 'LightGBM'))
    model_types.extend([
        ('random_forest', 'Random Forest'),
        ('gradient_boosting', 'Gradient Boosting'),
        ('logistic', 'Logistic Regression')
    ])
    
    results = {}
    models_trained = {}
    
    # Train each model
    for model_type, display_name in model_types:
        print(f"\n{'='*70}")
        print(f"🔧 Training: {display_name}")
        print(f"{'='*70}")
        
        try:
            # Create and train model
            model = create_model(model_type)
            print(f"Training {display_name}...")
            
            if model_type in ['xgboost', 'lightgbm']:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            else:
                model.fit(X_train, y_train)
            
            print(f"✅ {display_name} trained!")
            
            # Find optimal threshold on validation set
            optimal_threshold = find_optimal_threshold(model, X_val, y_val, target_recall=0.7)
            
            # Predict on test set with optimal threshold
            y_scores = model.predict_proba(X_test)[:, 1]
            y_pred = (y_scores >= optimal_threshold).astype(int)
            
            # Evaluate
            results[display_name] = evaluate_model(y_test, y_pred, y_scores, display_name)
            models_trained[display_name] = {'model': model, 'threshold': optimal_threshold}
            
        except Exception as e:
            print(f"❌ Error training {display_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not results:
        print("\n❌ No models were successfully trained!")
        return
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'models' / 'improved'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Rankings
    sorted_models = sorted(results.items(), key=lambda x: x[1]['recall'], reverse=True)
    print(f"\n{'='*70}")
    print(f"🏆 FINAL RANKINGS (by Recall)")
    print(f"{'='*70}")
    
    for rank, (model_name, metrics) in enumerate(sorted_models, 1):
        print(f"\n{rank}. {model_name}")
        print(f"   📊 Recall: {metrics['recall']:.1%} | PR AUC: {metrics['pr_auc']:.3f} | F1: {metrics['f1_score']:.3f}")
        print(f"   ✅ Catches {metrics['true_positives']}/{metrics['true_positives'] + metrics['false_negatives']} defaults")
        print(f"   ⚠️  {metrics['false_positives']} false alarms")
        
        # Save top 3 models
        if rank <= 3:
            model_info = models_trained[model_name]
            model_path = output_dir / f'rank{rank}_{model_name.replace(" ", "_").lower()}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model_info, f)
            print(f"   💾 Saved: {model_path.name}")
    
    # Plot comparison
    plot_comparison(results, y_test, output_dir)
    
    # Final recommendation
    best_model_name, best_metrics = sorted_models[0]
    print(f"\n{'='*70}")
    print(f"✨ RECOMMENDATION")
    print(f"{'='*70}")
    print(f"🥇 Best Model: {best_model_name}")
    print(f"\n📈 Performance:")
    print(f"   • Recall: {best_metrics['recall']:.1%} (catches {best_metrics['recall']*100:.0f}% of defaults)")
    print(f"   • PR AUC: {best_metrics['pr_auc']:.3f}")
    print(f"   • F1-Score: {best_metrics['f1_score']:.3f}")
    print(f"   • Defaults caught: {best_metrics['true_positives']}/{best_metrics['true_positives'] + best_metrics['false_negatives']}")
    print(f"   • Defaults missed: {best_metrics['false_negatives']}/{best_metrics['true_positives'] + best_metrics['false_negatives']}")
    
    print(f"\n🔄 Comparison with Autoencoder:")
    print(f"   • Autoencoder Recall: 2.5% (FAILED)")
    print(f"   • {best_model_name} Recall: {best_metrics['recall']:.1%}")
    print(f"   • Improvement: {(best_metrics['recall'] - 0.025) / 0.025 * 100:.0f}x better!")
    
    if best_metrics['recall'] >= 0.7:
        print(f"\n   ✅ This model is PRODUCTION READY!")
        print(f"   ✅ Recommended for real-world credit risk assessment")
    elif best_metrics['recall'] >= 0.6:
        print(f"\n   ✅ This model is GOOD and usable for production")
    else:
        print(f"\n   ⚠️  This model could be further improved")
    
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
