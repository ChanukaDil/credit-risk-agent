
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc, 
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)
from collections import Counter
from typing import Dict, Tuple, Optional
import os
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Optional: MLflow for experiment tracking
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️ MLflow not available. Install with: pip install mlflow")


class CreditRiskAutoencoder(nn.Module):
    """
    Deep Autoencoder for credit risk anomaly detection
    Detects unusual patterns that may indicate high default risk
    """
    
    def __init__(self, input_dim: int, encoding_dims: list = None):
        """
        Initialize autoencoder
        
        Args:
            input_dim: Number of input features
            encoding_dims: List of dimensions for encoder layers
        """
        super(CreditRiskAutoencoder, self).__init__()
        
        if encoding_dims is None:
            # Default architecture: gradually compress information
            encoding_dims = [64, 32, 16, 8]
        
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        
        # ═══════════════════════════════════════════════
        # ENCODER: Compress input to latent representation
        # ═══════════════════════════════════════════════
        encoder_layers = []
        prev_dim = input_dim
        
        for dim in encoding_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # ═══════════════════════════════════════════════
        # DECODER: Reconstruct input from latent representation
        # ═══════════════════════════════════════════════
        decoder_layers = []
        reversed_dims = list(reversed(encoding_dims[:-1])) + [input_dim]
        prev_dim = encoding_dims[-1]
        
        for i, dim in enumerate(reversed_dims):
            decoder_layers.append(nn.Linear(prev_dim, dim))
            if i < len(reversed_dims) - 1:
                decoder_layers.extend([
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
            prev_dim = dim
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        """Forward pass: encode then decode"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_encoding(self, x):
        """Get latent representation (compressed features)"""
        return self.encoder(x)


class AutoencoderTrainer:
    """
    Complete training and evaluation pipeline for autoencoder
    Integrated with your CreditRiskPreprocessor
    """
    
    def __init__(
        self,
        input_dim: int,
        encoding_dims: list = None,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        device: str = None
    ):
        """
        Initialize trainer
        
        Args:
            input_dim: Number of input features
            encoding_dims: Encoder layer dimensions
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims or [64, 32, 16, 8]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Set device (GPU if available)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🖥️  Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        # Initialize model
        self.model = CreditRiskAutoencoder(input_dim, encoding_dims).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # Threshold for anomaly detection
        self.threshold = None
        
        print(f"✅ Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def load_preprocessed_data(
        self, 
        dataset_name: str = 'lending_club',
        data_dir: str = 'data/processed'
    ) -> Dict[str, np.ndarray]:
        """
        Load preprocessed data from your pipeline
        
        Args:
            dataset_name: Name of the processed dataset
            data_dir: Directory containing processed data
            
        Returns:
            Dictionary with all splits
        """
        print(f"\n{'='*70}")
        print(f"LOADING PREPROCESSED DATA: {dataset_name}")
        print(f"{'='*70}")
        
        # Check if data is in subdirectory or directly in processed folder
        data_path = Path(data_dir) / dataset_name
        if not data_path.exists():
            # Try direct path (for your case)
            data_path = Path(data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(
                f"Processed data not found at: {data_path}\n"
                f"Please run preprocessing first:\n"
                f"  python src/run_preprocessing.py"
            )
        
        # Load all splits
        splits = {}
        for split_name in ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']:
            file_path = data_path / f"{split_name}.npy"
            if 'y_' in split_name:
                # Load target arrays with pickle=True (they contain strings)
                splits[split_name] = np.load(file_path, allow_pickle=True)
                # Convert string labels to numeric
                splits[split_name] = (splits[split_name] == 'YES').astype(int)
            else:
                splits[split_name] = np.load(file_path)
            print(f"✅ Loaded {split_name}: {splits[split_name].shape}")
        
        # Load metadata if available
        metadata_path = data_path / "metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"\n📊 Dataset Metadata:")
            print(f"   Processed date: {metadata.get('processed_date', 'N/A')}")
        
        print(f"\n✅ All data loaded successfully!")
        
        return splits
    
    def prepare_dataloaders(
        self, 
        data_splits: Dict[str, np.ndarray],
        train_on_normal_only: bool = True
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare PyTorch DataLoaders from numpy arrays
        
        Args:
            data_splits: Dictionary with X_train, y_train, etc.
            train_on_normal_only: If True, train only on non-default cases (class 0)
            
        Returns:
            train_loader, val_loader, test_loader
        """
        print(f"\n{'='*70}")
        print("PREPARING DATALOADERS")
        print(f"{'='*70}")
        
        X_train = data_splits['X_train']
        y_train = data_splits['y_train']
        
        X_val = data_splits['X_val']
        y_val = data_splits['y_val']
        
        X_test = data_splits['X_test']
        y_test = data_splits['y_test']
        
        # For autoencoder anomaly detection:
        # - Train ONLY on normal cases (non-defaults, class 0)
        # - Model learns what "normal" looks like
        # - Defaults (class 1) will have high reconstruction error
        
        if train_on_normal_only:
            print("\n🎯 Training Strategy: ANOMALY DETECTION")
            print("   Training only on non-default cases (class 0)")
            print("   Model will learn 'normal' patterns")
            print("   Defaults will have high reconstruction error\n")
            
            X_train_normal = X_train[y_train == 0]
            X_val_normal = X_val[y_val == 0]
            
            print(f"📊 Training Data:")
            print(f"   Original: {len(X_train):,} samples")
            print(f"   Normal only: {len(X_train_normal):,} samples ({len(X_train_normal)/len(X_train)*100:.1f}%)")
            print(f"   Excluded defaults: {len(X_train) - len(X_train_normal):,}")
            
            print(f"\n📊 Validation Data:")
            print(f"   Original: {len(X_val):,} samples")
            print(f"   Normal only: {len(X_val_normal):,} samples ({len(X_val_normal)/len(X_val)*100:.1f}%)")
        else:
            print("\n🎯 Training Strategy: RECONSTRUCTION")
            print("   Training on all cases")
            X_train_normal = X_train
            X_val_normal = X_val
        
        print(f"\n📊 Test Data (Evaluation):")
        print(f"   Total: {len(X_test):,} samples")
        print(f"   Non-defaults: {sum(y_test == 0):,} ({sum(y_test == 0)/len(y_test)*100:.1f}%)")
        print(f"   Defaults: {sum(y_test == 1):,} ({sum(y_test == 1)/len(y_test)*100:.1f}%)")
        
        # Convert to PyTorch tensors
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_normal),
            torch.FloatTensor(X_train_normal)  # Target is same as input
        )
        
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_normal),
            torch.FloatTensor(X_val_normal)
        )
        
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test)  # Keep labels for evaluation
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        print(f"\n✅ DataLoaders created:")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_x, batch_target in progress_bar:
            batch_x = batch_x.to(self.device)
            batch_target = batch_target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_target in val_loader:
                batch_x = batch_x.to(self.device)
                batch_target = batch_target.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_target)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        use_mlflow: bool = False
    ):
        """
        Complete training loop with early stopping
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            early_stopping_patience: Stop if no improvement for N epochs
            use_mlflow: Use MLflow for experiment tracking
        """
        print(f"\n{'='*70}")
        print("STARTING AUTOENCODER TRAINING")
        print(f"{'='*70}")
        print(f"Epochs: {epochs}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"{'='*70}\n")
        
        if use_mlflow and MLFLOW_AVAILABLE:
            mlflow.set_experiment("credit_risk_autoencoder")
            mlflow.start_run()
            
            # Log hyperparameters
            mlflow.log_params({
                'input_dim': self.input_dim,
                'encoding_dims': str(self.encoding_dims),
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': epochs,
                'device': str(self.device)
            })
        
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Record history
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Log to MLflow
            if use_mlflow and MLFLOW_AVAILABLE:
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }, step=epoch)
            
            # Print progress
            print(f"Epoch [{epoch+1:3d}/{epochs}] - "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}", end='')
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(" ✓ Best model!")
            else:
                patience_counter += 1
                print(f" (patience: {patience_counter}/{early_stopping_patience})")
            
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠️ Early stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        
        print(f"\n{'='*70}")
        print(f"✅ Training completed!")
        print(f"   Best validation loss: {self.best_val_loss:.6f}")
        print(f"   Total epochs: {len(self.history['epoch'])}")
        print(f"{'='*70}\n")
        
        if use_mlflow and MLFLOW_AVAILABLE:
            mlflow.log_metric('best_val_loss', self.best_val_loss)
            mlflow.pytorch.log_model(self.model, "model")
            mlflow.end_run()
    
    def calculate_reconstruction_errors(self, data_loader: DataLoader) -> np.ndarray:
        """
        Calculate reconstruction errors for each sample
        High error = Anomaly (potential default)
        """
        self.model.eval()
        errors = []
        
        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                
                # MSE per sample (mean across features)
                batch_errors = torch.mean((batch_x - outputs) ** 2, dim=1)
                errors.extend(batch_errors.cpu().numpy())
        
        return np.array(errors)
    
    def determine_threshold(
        self,
        val_loader: DataLoader,
        percentile: float = 95
    ) -> float:
        """
        Determine anomaly threshold based on validation set
        
        Args:
            val_loader: Validation data (normal cases only)
            percentile: Percentile for threshold (95 = 95th percentile)
            
        Returns:
            Threshold value
        """
        print(f"\n{'='*70}")
        print("DETERMINING ANOMALY THRESHOLD")
        print(f"{'='*70}")
        print(f"Using {percentile}th percentile of reconstruction errors")
        print("on normal (non-default) validation cases\n")
        
        errors = self.calculate_reconstruction_errors(val_loader)
        threshold = np.percentile(errors, percentile)
        
        self.threshold = threshold
        
        print(f"📊 Validation Error Statistics:")
        print(f"   Mean: {np.mean(errors):.6f}")
        print(f"   Std: {np.std(errors):.6f}")
        print(f"   Min: {np.min(errors):.6f}")
        print(f"   Max: {np.max(errors):.6f}")
        print(f"   {percentile}th percentile: {threshold:.6f}")
        
        print(f"\n✅ Threshold set to: {threshold:.6f}")
        print(f"   Samples with error > {threshold:.6f} will be flagged as anomalies")
        print(f"{'='*70}\n")
        
        return threshold
    
    def evaluate(
        self,
        test_loader: DataLoader,
        threshold: float = None
    ) -> Dict:
        """
        Evaluate model on test set
        
        Args:
            test_loader: Test data loader
            threshold: Anomaly threshold (uses self.threshold if None)
            
        Returns:
            Dictionary with metrics and results
        """
        if threshold is None:
            threshold = self.threshold
            
        if threshold is None:
            raise ValueError("Threshold not set! Run determine_threshold() first.")
        
        print(f"\n{'='*70}")
        print("EVALUATING MODEL ON TEST SET")
        print(f"{'='*70}")
        print(f"Threshold: {threshold:.6f}\n")
        
        self.model.eval()
        all_errors = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(test_loader, desc="Evaluating"):
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                
                errors = torch.mean((batch_x - outputs) ** 2, dim=1)
                all_errors.extend(errors.cpu().numpy())
                all_labels.extend(batch_y.numpy())
        
        all_errors = np.array(all_errors)
        all_labels = np.array(all_labels).astype(int)
        
        # Predictions: error > threshold → anomaly (default = 1)
        predictions = (all_errors > threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, predictions),
            'precision': precision_score(all_labels, predictions, zero_division=0),
            'recall': recall_score(all_labels, predictions, zero_division=0),
            'f1': f1_score(all_labels, predictions, zero_division=0),
            'roc_auc': roc_auc_score(all_labels, all_errors)
        }
        
        print("📊 PERFORMANCE METRICS:")
        print("-" * 40)
        for metric, value in metrics.items():
            print(f"{metric.upper():12s}: {value:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, predictions)
        print(f"\n📊 CONFUSION MATRIX:")
        print("-" * 40)
        print(cm)
        print(f"\nTrue Negatives (TN):  {cm[0,0]:,} - Correctly identified non-defaults")
        print(f"False Positives (FP): {cm[0,1]:,} - Normal cases flagged as defaults")
        print(f"False Negatives (FN): {cm[1,0]:,} - Defaults missed")
        print(f"True Positives (TP):  {cm[1,1]:,} - Correctly caught defaults")
        
        # Classification report
        print(f"\n📊 CLASSIFICATION REPORT:")
        print("-" * 40)
        print(classification_report(all_labels, predictions, 
                                   target_names=['Non-Default (0)', 'Default (1)']))
        
        print(f"{'='*70}\n")
        
        return {
            'metrics': metrics,
            'errors': all_errors,
            'labels': all_labels,
            'predictions': predictions,
            'confusion_matrix': cm,
            'threshold': threshold
        }
    
    def plot_training_history(self, save_path: str = None):
        """Plot training and validation loss"""
        plt.figure(figsize=(12, 6))
        
        plt.plot(self.history['epoch'], self.history['train_loss'], 
                label='Train Loss', marker='o', linewidth=2, markersize=4)
        plt.plot(self.history['epoch'], self.history['val_loss'], 
                label='Validation Loss', marker='s', linewidth=2, markersize=4)
        
        # Mark best epoch
        best_epoch = np.argmin(self.history['val_loss']) + 1
        plt.axvline(best_epoch, color='red', linestyle='--', 
                   label=f'Best Epoch ({best_epoch})', alpha=0.7)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.title('Autoencoder Training History', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        
        plt.show()
    
    def plot_error_distribution(
        self,
        errors: np.ndarray,
        labels: np.ndarray,
        threshold: float,
        save_path: str = None
    ):
        """Plot reconstruction error distribution"""
        plt.figure(figsize=(14, 6))
        
        # Separate errors by class
        normal_errors = errors[labels == 0]
        default_errors = errors[labels == 1]
        
        # Plot histograms
        plt.hist(normal_errors, bins=50, alpha=0.6, label=f'Non-Default ({len(normal_errors):,})', 
                color='blue', edgecolor='black')
        plt.hist(default_errors, bins=50, alpha=0.6, label=f'Default ({len(default_errors):,})', 
                color='red', edgecolor='black')
        plt.axvline(threshold, color='green', linestyle='--', 
                   linewidth=3, label=f'Threshold ({threshold:.4f})')
        
        # Add statistics
        plt.text(0.02, 0.98, 
                f'Non-Default:\n  Mean: {np.mean(normal_errors):.4f}\n  Std: {np.std(normal_errors):.4f}',
                transform=plt.gca().transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='blue', alpha=0.2),
                fontsize=9)
        
        plt.text(0.98, 0.98, 
                f'Default:\n  Mean: {np.mean(default_errors):.4f}\n  Std: {np.std(default_errors):.4f}',
                transform=plt.gca().transAxes, 
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.2),
                fontsize=9)
        
        plt.xlabel('Reconstruction Error', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Reconstruction Error Distribution', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        
        plt.show()
    
    def plot_roc_curve(
        self,
        labels: np.ndarray,
        errors: np.ndarray,
        save_path: str = None
    ):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(labels, errors)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Credit Risk Anomaly Detection', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        
        plt.show()
    
    def save_model(self, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'threshold': self.threshold,
            'input_dim': self.input_dim,
            'encoding_dims': self.encoding_dims,
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f"💾 Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.threshold = checkpoint.get('threshold')
        self.history = checkpoint.get('history', self.history)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"✅ Model loaded from: {filepath}")
        if self.threshold:
            print(f"   Threshold: {self.threshold:.6f}")


def train_autoencoder(
    dataset_name: str = 'lending_club',
    data_dir: str = 'data/processed',
    encoding_dims: list = None,
    learning_rate: float = 0.001,
    batch_size: int = 256,
    epochs: int = 50,
    early_stopping_patience: int = 10,
    threshold_percentile: float = 95,
    use_mlflow: bool = False,
    save_visualizations: bool = True
) -> Tuple[AutoencoderTrainer, Dict]:
    """
    Complete training pipeline
    
    Args:
        dataset_name: Name of processed dataset folder
        data_dir: Directory containing processed data
        encoding_dims: Encoder architecture
        learning_rate: Learning rate
        batch_size: Batch size
        epochs: Maximum epochs
        early_stopping_patience: Early stopping patience
        threshold_percentile: Percentile for threshold
        use_mlflow: Use MLflow tracking
        save_visualizations: Save plots
        
    Returns:
        trainer, evaluation_results
    """
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║         AUTOENCODER TRAINING FOR CREDIT RISK DETECTION          ║
║              Anomaly Detection Approach                          ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Load preprocessed data
    trainer = AutoencoderTrainer(
        input_dim=1,  # Placeholder, will be updated
        encoding_dims=encoding_dims,
        learning_rate=learning_rate,
        batch_size=batch_size
    )
    
    splits = trainer.load_preprocessed_data(dataset_name, data_dir)
    
    # Update input_dim based on actual data
    input_dim = splits['X_train'].shape[1]
    trainer = AutoencoderTrainer(
        input_dim=input_dim,
        encoding_dims=encoding_dims,
        learning_rate=learning_rate,
        batch_size=batch_size
    )
    
    # Step 2: Prepare dataloaders
    train_loader, val_loader, test_loader = trainer.prepare_dataloaders(
        splits, 
        train_on_normal_only=True
    )
    
    # Step 3: Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        use_mlflow=use_mlflow
    )
    
    # Step 4: Determine threshold
    trainer.determine_threshold(val_loader, percentile=threshold_percentile)
    
    # Step 4.5: Save validation errors for risk scoring calibration
    print("\n💾 Saving validation errors for risk scoring...")
    val_errors = trainer.calculate_reconstruction_errors(val_loader)
    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)
    np.save(results_dir / 'validation_errors.npy', val_errors)
    print(f"✅ Saved: {results_dir / 'validation_errors.npy'}")
    
    # Step 5: Evaluate
    results = trainer.evaluate(test_loader)
    
    # Step 6: Save visualizations
    if save_visualizations:
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70 + "\n")
        
        viz_dir = Path('results/figures/autoencoder')
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        trainer.plot_training_history(str(viz_dir / 'training_history.png'))
        trainer.plot_error_distribution(
            results['errors'], 
            results['labels'],
            trainer.threshold,
            str(viz_dir / 'error_distribution.png')
        )
        trainer.plot_roc_curve(
            results['labels'],
            results['errors'],
            str(viz_dir / 'roc_curve.png')
        )
    
    # Step 7: Save model
    model_dir = Path('models/autoencoder')
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f'{dataset_name}_autoencoder.pth'
    trainer.save_model(str(model_path))
    
    print("\n" + "="*70)
    print("✅ TRAINING PIPELINE COMPLETE!")
    print("="*70)
    print(f"\n📁 Outputs saved to:")
    print(f"   Model: {model_path}")
    print(f"   Figures: results/figures/autoencoder/")
    print(f"\n🎯 Model Performance:")
    print(f"   ROC-AUC: {results['metrics']['roc_auc']:.4f}")
    print(f"   F1-Score: {results['metrics']['f1']:.4f}")
    print(f"   Recall: {results['metrics']['recall']:.4f}")
    print(f"   Precision: {results['metrics']['precision']:.4f}")
    
    return trainer, results


if __name__ == "__main__":
    """
    Main execution
    Usage: python src/autoencoder_training.py
    """
    
    # Train autoencoder
    trainer, results = train_autoencoder(
        dataset_name='default',  # Updated to match your processed data
        data_dir='data/processed',
        encoding_dims=[64, 32, 16, 8],  # Architecture
        learning_rate=0.001,
        batch_size=256,
        epochs=50,
        early_stopping_patience=10,
        threshold_percentile=95,
        use_mlflow=False,  # Set to True if you have MLflow
        save_visualizations=True
    )
    
    print("\n🎉 All done! Your autoencoder is trained and ready to use!")
    print("\nNext steps:")
    print("1. Check results/figures/autoencoder/ for visualizations")
    print("2. Use the saved model for anomaly detection on new data")
    print("3. Experiment with different architectures and hyperparameters")