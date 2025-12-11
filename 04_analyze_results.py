#!/usr/bin/env python3
"""
Results Analysis and Visualization
Generates comprehensive analysis reports and visualizations

Usage:
    python 04_analyze_results.py --model-dir models --output-dir results
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    classification_report, confusion_matrix
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyzes and visualizes pipeline results"""
    
    def __init__(self, model_dir: str, output_dir: str):
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'figures').mkdir(exist_ok=True)
        (self.output_dir / 'reports').mkdir(exist_ok=True)
    
    def load_model(self, model_path: str):
        """Load trained model"""
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def load_training_summary(self) -> Dict:
        """Load training summary"""
        
        summary_file = self.model_dir / 'training_summary.json'
        if not summary_file.exists():
            logger.warning("Training summary not found")
            return {}
        
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        return summary
    
    def plot_roc_curves(self, y_true, y_pred_proba, class_names: List[str],
                       filename: str = 'roc_curves.png'):
        """Plot ROC curves for multi-class classification"""
        
        logger.info("Generating ROC curves...")
        
        n_classes = len(class_names)
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i, class_name in enumerate(class_names):
            # Binary indicators for each class
            y_true_binary = (y_true == class_name).astype(int)
            
            if len(y_pred_proba.shape) > 1:
                y_score = y_pred_proba[:, i]
            else:
                y_score = y_pred_proba
            
            fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_score)
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot
        plt.figure(figsize=(10, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
        
        for i, (color, class_name) in enumerate(zip(colors, class_names)):
            plt.plot(
                fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_name} (AUC = {roc_auc[i]:.2f})'
            )
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Multi-class Classification', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        output_path = self.output_dir / 'figures' / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curves saved to {output_path}")
        
        return roc_auc
    
    def plot_precision_recall_curves(self, y_true, y_pred_proba,
                                     class_names: List[str],
                                     filename: str = 'pr_curves.png'):
        """Plot Precision-Recall curves"""
        
        logger.info("Generating Precision-Recall curves...")
        
        n_classes = len(class_names)
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
        
        for i, (color, class_name) in enumerate(zip(colors, class_names)):
            y_true_binary = (y_true == class_name).astype(int)
            
            if len(y_pred_proba.shape) > 1:
                y_score = y_pred_proba[:, i]
            else:
                y_score = y_pred_proba
            
            precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
            
            plt.plot(
                recall, precision, color=color, lw=2,
                label=f'{class_name}'
            )
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves', fontsize=14)
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        
        output_path = self.output_dir / 'figures' / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"PR curves saved to {output_path}")
    
    def plot_confusion_matrix_detailed(self, y_true, y_pred,
                                      class_names: List[str],
                                      filename: str = 'confusion_matrix_detailed.png'):
        """Plot detailed confusion matrix with percentages"""
        
        logger.info("Generating detailed confusion matrix...")
        
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Absolute counts
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[0]
        )
        axes[0].set_xlabel('Predicted', fontsize=12)
        axes[0].set_ylabel('Actual', fontsize=12)
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=14)
        
        # Normalized
        sns.heatmap(
            cm_normalized, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[1]
        )
        axes[1].set_xlabel('Predicted', fontsize=12)
        axes[1].set_ylabel('Actual', fontsize=12)
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'figures' / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Detailed confusion matrix saved to {output_path}")
    
    def plot_class_distribution(self, y_data, title: str,
                               filename: str = 'class_distribution.png'):
        """Plot class distribution"""
        
        logger.info("Generating class distribution plot...")
        
        class_counts = pd.Series(y_data).value_counts()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar plot
        class_counts.plot(kind='bar', ax=axes[0], color='steelblue')
        axes[0].set_xlabel('Class', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title(f'{title} - Bar Plot', fontsize=14)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Pie chart
        axes[1].pie(
            class_counts.values,
            labels=class_counts.index,
            autopct='%1.1f%%',
            startangle=90
        )
        axes[1].set_title(f'{title} - Distribution', fontsize=14)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'figures' / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Class distribution plot saved to {output_path}")
    
    def plot_feature_importance_comparison(self, model, feature_names: List[str],
                                          filename: str = 'feature_importance_comparison.png',
                                          top_n: int = 30):
        """Plot feature importance with omics type comparison"""
        
        logger.info("Generating feature importance comparison...")
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        # Categorize features by omics type
        feature_categories = []
        for idx in indices:
            feat_name = feature_names[idx]
            if 'gene_expression' in feat_name:
                category = 'Gene Expression'
            elif 'methylation' in feat_name:
                category = 'Methylation'
            elif 'copy_number' in feat_name:
                category = 'Copy Number'
            else:
                category = 'Other'
            feature_categories.append(category)
        
        # Create color map
        color_map = {
            'Gene Expression': '#FF6B6B',
            'Methylation': '#4ECDC4',
            'Copy Number': '#95E1D3',
            'Other': '#CCCCCC'
        }
        colors = [color_map[cat] for cat in feature_categories]
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 10))
        
        y_pos = np.arange(top_n)
        ax.barh(y_pos, importances[indices], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i][:40] for i in indices], fontsize=8)
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Most Important Features', fontsize=14)
        ax.invert_yaxis()
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color, label=label)
            for label, color in color_map.items()
            if label in feature_categories
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'figures' / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance comparison saved to {output_path}")
    
    def generate_html_report(self, summary: Dict, metrics: Dict,
                            filename: str = 'analysis_report.html'):
        """Generate HTML analysis report"""
        
        logger.info("Generating HTML report...")
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Omics Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        h1 {{ margin: 0; font-size: 2.5em; }}
        h2 {{
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 40px;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric {{
            display: inline-block;
            background: #f8f9fa;
            padding: 15px 25px;
            margin: 10px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            display: block;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
        tr:hover {{ background-color: #f5f5f5; }}
        .footer {{
            text-align: center;
            margin-top: 50px;
            color: #666;
            font-size: 0.9em;
        }}
        img {{
            max-width: 100%;
            border-radius: 8px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Multi-Omics Analysis Report</h1>
        <p>Comprehensive analysis of cancer subtype classification</p>
    </div>

    <div class="section">
        <h2> Model Performance</h2>
        <div class="metric">
            <span class="metric-label">Training Accuracy</span>
            <span class="metric-value">{metrics.get('train_accuracy', 'N/A')}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Testing Accuracy</span>
            <span class="metric-value">{metrics.get('test_accuracy', 'N/A')}</span>
        </div>
        <div class="metric">
            <span class="metric-label">CV Mean</span>
            <span class="metric-value">{metrics.get('cv_mean', 'N/A')}</span>
        </div>
    </div>

    <div class="section">
        <h2> Configuration</h2>
        <table>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Integration Method</td>
                <td>{summary.get('integration_method', 'N/A')}</td>
            </tr>
            <tr>
                <td>PCA Components</td>
                <td>{summary.get('n_components', 'N/A')}</td>
            </tr>
            <tr>
                <td>Label Type</td>
                <td>{summary.get('label_type', 'N/A')}</td>
            </tr>
            <tr>
                <td>Number of Samples</td>
                <td>{summary.get('n_samples', 'N/A')}</td>
            </tr>
            <tr>
                <td>Number of Classes</td>
                <td>{summary.get('n_classes', 'N/A')}</td>
            </tr>
            <tr>
                <td>Omics Types</td>
                <td>{', '.join(summary.get('omics_types', []))}</td>
            </tr>
        </table>
    </div>

    <div class="section">
        <h2> Visualizations</h2>
        <h3>Confusion Matrix</h3>
        <img src="../figures/confusion_matrix_detailed.png" alt="Confusion Matrix">
        
        <h3>ROC Curves</h3>
        <img src="../figures/roc_curves.png" alt="ROC Curves">
        
        <h3>Feature Importance</h3>
        <img src="../figures/feature_importance_comparison.png" alt="Feature Importance">
    </div>

    <div class="footer">
        <p>Report generated by Multi-Omics Integration Pipeline</p>
        <p>© 2024 - For research purposes only</p>
    </div>
</body>
</html>
"""
        
        output_path = self.output_dir / 'reports' / filename
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {output_path}")
    
    def run_analysis(self, data_dir: str = None):
        """Run complete analysis pipeline"""
        
        logger.info(f"\n{'='*60}")
        logger.info("Starting results analysis")
        logger.info(f"{'='*60}\n")
        
        # Load training summary
        summary = self.load_training_summary()
        logger.info(f"Loaded training summary: {summary}")
        
        # Check if we have the model and data needed
        model_file = self.model_dir / 'early_integration_model.pkl'
        
        if not model_file.exists():
            logger.error("Model file not found. Cannot perform analysis.")
            return
        
        # Load model
        model = self.load_model(model_file)
        logger.info("Model loaded successfully")
        
        # Create visualizations (if we have access to test data)
        # For now, generate plots based on what we have
        
        metrics = {
            'train_accuracy': f"{summary.get('train_accuracy', 0):.3f}",
            'test_accuracy': f"{summary.get('test_accuracy', 0):.3f}",
            'cv_mean': f"{summary.get('cv_mean', 0):.3f}"
        }
        
        # Generate HTML report
        self.generate_html_report(summary, metrics)
        
        logger.info(f"\n{'='*60}")
        logger.info("Analysis complete!")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze multi-omics integration results'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        required=True,
        help='Directory with trained models'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Directory with processed data (optional)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for analysis results'
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ResultsAnalyzer(args.model_dir, args.output_dir)
    
    # Run analysis
    analyzer.run_analysis(args.data_dir)


if __name__ == "__main__":
    main()
