#!/usr/bin/env python3
"""
Multi-Omics Integration and Model Training
Trains models for cancer subtype classification using integrated omics data

Usage:
    python 03_train_model.py --data-dir data/processed --output-dir models
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiOmicsIntegrator:
    """Integrates and analyzes multi-omics data"""
    
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.omics_data = {}
        self.clinical_data = None
        self.labels = None
        
    def load_data(self):
        """Load processed omics data"""
        
        logger.info("Loading processed data...")
        
        # Load gene expression
        gene_file = self.data_dir / "gene_expression_processed.csv"
        if gene_file.exists():
            self.omics_data['gene_expression'] = pd.read_csv(gene_file, index_col=0)
            logger.info(f"Gene expression: {self.omics_data['gene_expression'].shape}")
        
        # Load methylation
        methyl_file = self.data_dir / "methylation_processed.csv"
        if methyl_file.exists():
            self.omics_data['methylation'] = pd.read_csv(methyl_file, index_col=0)
            logger.info(f"Methylation: {self.omics_data['methylation'].shape}")
        
        # Load copy number
        cnv_file = self.data_dir / "copy_number_processed.csv"
        if cnv_file.exists():
            self.omics_data['copy_number'] = pd.read_csv(cnv_file, index_col=0)
            logger.info(f"Copy number: {self.omics_data['copy_number'].shape}")
        
        # Load clinical
        clinical_file = self.data_dir / "clinical_processed.csv"
        if clinical_file.exists():
            self.clinical_data = pd.read_csv(clinical_file)
            logger.info(f"Clinical data: {self.clinical_data.shape}")
        
        if not self.omics_data:
            raise ValueError("No omics data found!")
    
    def prepare_labels(self, label_type: str = 'tumor_stage'):
        """Prepare classification labels from clinical data"""
        
        logger.info(f"Preparing labels based on: {label_type}")
        
        if self.clinical_data is None:
            logger.warning("No clinical data available")
            return None
        
        # Get common samples
        sample_ids = list(self.omics_data.values())[0].columns
        
        # Map sample IDs to labels
        labels = {}
        for sample_id in sample_ids:
            # Match sample ID to clinical data (handling TCGA barcode format)
            matching_rows = self.clinical_data[
                self.clinical_data['submitter_id'].str.contains(sample_id[:12], na=False)
            ]
            
            if not matching_rows.empty:
                label_value = matching_rows.iloc[0][label_type]
                if pd.notna(label_value):
                    labels[sample_id] = str(label_value)
        
        if not labels:
            logger.warning(f"No labels found for {label_type}")
            return None
        
        self.labels = pd.Series(labels)
        logger.info(f"Labels prepared: {len(self.labels)} samples")
        logger.info(f"Label distribution:\n{self.labels.value_counts()}")
        
        return self.labels
    
    def reduce_dimensions_pca(self, n_components: int = 50):
        """Reduce dimensionality using PCA"""
        
        logger.info(f"Applying PCA (n_components={n_components})...")
        
        reduced_data = {}
        
        for omics_name, data in self.omics_data.items():
            logger.info(f"Processing {omics_name}...")
            
            # Transpose so samples are rows
            X = data.T
            
            # Apply PCA
            n_comp = min(n_components, X.shape[0], X.shape[1])
            pca = PCA(n_components=n_comp)
            X_reduced = pca.fit_transform(X)
            
            # Create DataFrame
            columns = [f"{omics_name}_PC{i+1}" for i in range(n_comp)]
            df_reduced = pd.DataFrame(
                X_reduced,
                index=data.columns,
                columns=columns
            )
            
            reduced_data[omics_name] = df_reduced
            
            logger.info(f"  Original shape: {X.shape}")
            logger.info(f"  Reduced shape: {X_reduced.shape}")
            logger.info(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        return reduced_data
    
    def early_integration(self, reduced_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Early integration: concatenate all omics"""
        
        logger.info("Performing early integration...")
        
        # Concatenate along columns (features)
        integrated = pd.concat(reduced_data.values(), axis=1)
        
        logger.info(f"Integrated shape: {integrated.shape}")
        return integrated
    
    def late_integration(self, reduced_data: Dict[str, pd.DataFrame],
                        labels: pd.Series) -> Dict:
        """Late integration: train separate models then ensemble"""
        
        logger.info("Performing late integration...")
        
        models = {}
        predictions = {}
        
        # Find common samples
        common_samples = set(labels.index)
        for df in reduced_data.values():
            common_samples = common_samples.intersection(set(df.index))
        common_samples = sorted(list(common_samples))
        
        y = labels[common_samples]
        
        # Train model for each omics
        for omics_name, data in reduced_data.items():
            logger.info(f"Training {omics_name} model...")
            
            X = data.loc[common_samples]
            
            # Train model
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            clf.fit(X, y)
            
            # Store model and predictions
            models[omics_name] = clf
            predictions[omics_name] = clf.predict_proba(X)
        
        return {'models': models, 'predictions': predictions}
    
    def train_ensemble(self, X: pd.DataFrame, y: pd.Series,
                      test_size: float = 0.2) -> Dict:
        """Train ensemble classifier"""
        
        logger.info("Training ensemble classifier...")
        
        # Find common samples
        common_samples = list(set(X.index).intersection(set(y.index)))
        X_common = X.loc[common_samples]
        y_common = y[common_samples]
        
        logger.info(f"Training samples: {len(common_samples)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_common, y_common, test_size=test_size, random_state=42, stratify=y_common
        )
        
        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Train Random Forest
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        
        logger.info("Training model...")
        clf.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        
        # Evaluate
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        logger.info(f"Training accuracy: {train_acc:.3f}")
        logger.info(f"Testing accuracy: {test_acc:.3f}")
        
        # Cross-validation
        logger.info("Performing cross-validation...")
        cv_scores = cross_val_score(
            clf, X_train, y_train, cv=5, scoring='accuracy'
        )
        logger.info(f"CV scores: {cv_scores}")
        logger.info(f"CV mean: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        # Classification report
        logger.info("\nTest Set Classification Report:")
        print(classification_report(y_test, y_pred_test))
        
        results = {
            'model': clf,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'cv_scores': cv_scores,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred_test': y_pred_test,
            'feature_names': X.columns.tolist()
        }
        
        return results
    
    def save_model(self, model, filename: str):
        """Save trained model"""
        
        output_path = self.output_dir / filename
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {output_path}")
    
    def plot_confusion_matrix(self, y_true, y_pred, filename: str):
        """Plot confusion matrix"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {output_path}")
    
    def plot_feature_importance(self, model, feature_names: List[str],
                               filename: str, top_n: int = 20):
        """Plot feature importance"""
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(top_n), importances[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Features')
        plt.gca().invert_yaxis()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved to {output_path}")
    
    def run_pipeline(self, integration_method: str = 'early',
                    n_components: int = 50, label_type: str = 'tumor_stage'):
        """Run complete training pipeline"""
        
        logger.info(f"\n{'='*60}")
        logger.info("Starting training pipeline")
        logger.info(f"Integration method: {integration_method}")
        logger.info(f"{'='*60}\n")
        
        # Load data
        self.load_data()
        
        # Prepare labels
        labels = self.prepare_labels(label_type)
        if labels is None:
            logger.error("Cannot proceed without labels")
            return
        
        # Reduce dimensions
        reduced_data = self.reduce_dimensions_pca(n_components)
        
        # Integrate and train
        if integration_method == 'early':
            # Early integration
            X_integrated = self.early_integration(reduced_data)
            results = self.train_ensemble(X_integrated, labels)
            
            # Save model
            self.save_model(results['model'], 'early_integration_model.pkl')
            
            # Plot confusion matrix
            self.plot_confusion_matrix(
                results['y_test'],
                results['y_pred_test'],
                'confusion_matrix.png'
            )
            
            # Plot feature importance
            self.plot_feature_importance(
                results['model'],
                results['feature_names'],
                'feature_importance.png'
            )
            
        elif integration_method == 'late':
            # Late integration
            late_results = self.late_integration(reduced_data, labels)
            
            # Save models
            for omics_name, model in late_results['models'].items():
                self.save_model(model, f'late_integration_{omics_name}_model.pkl')
        
        # Save results summary
        summary = {
            'integration_method': integration_method,
            'n_components': n_components,
            'label_type': label_type,
            'n_samples': len(labels),
            'n_classes': len(labels.unique()),
            'omics_types': list(self.omics_data.keys())
        }
        
        if integration_method == 'early':
            summary['train_accuracy'] = float(results['train_acc'])
            summary['test_accuracy'] = float(results['test_acc'])
            summary['cv_mean'] = float(results['cv_scores'].mean())
            summary['cv_std'] = float(results['cv_scores'].std())
        
        with open(self.output_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info("Training complete!")
        logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Train multi-omics integration models'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory with processed data'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Output directory for trained models'
    )
    
    parser.add_argument(
        '--integration-method',
        type=str,
        choices=['early', 'late'],
        default='early',
        help='Integration method'
    )
    
    parser.add_argument(
        '--n-components',
        type=int,
        default=50,
        help='Number of PCA components'
    )
    
    parser.add_argument(
        '--label-type',
        type=str,
        default='tumor_stage',
        help='Clinical variable to use as label'
    )
    
    args = parser.parse_args()
    
    # Initialize integrator
    integrator = MultiOmicsIntegrator(args.data_dir, args.output_dir)
    
    # Run pipeline
    integrator.run_pipeline(
        integration_method=args.integration_method,
        n_components=args.n_components,
        label_type=args.label_type
    )


if __name__ == "__main__":
    main()
