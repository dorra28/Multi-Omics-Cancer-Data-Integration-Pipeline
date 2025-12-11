#!/usr/bin/env python3
"""
Data Preprocessing Pipeline
Preprocesses TCGA multi-omics data for integration and analysis

Usage:
    python 02_preprocess.py --input-dir data/raw --output-dir data/processed
"""

import argparse
import json
import logging
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import gzip

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiOmicsPreprocessor:
    """Preprocesses multi-omics data from TCGA"""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_gene_expression_files(self) -> pd.DataFrame:
        """Load and merge gene expression files"""
        
        logger.info("Loading gene expression data...")
        gene_expr_dir = self.input_dir / "gene_expression"
        
        if not gene_expr_dir.exists():
            logger.warning("Gene expression directory not found")
            return None
        
        files = list(gene_expr_dir.glob("*.tsv*"))
        logger.info(f"Found {len(files)} gene expression files")
        
        data_dict = {}
        
        for i, file_path in enumerate(files, 1):
            if i % 50 == 0:
                logger.info(f"Processing file {i}/{len(files)}")
            
            try:
                # Read file (handle gzipped files)
                if file_path.suffix == '.gz':
                    df = pd.read_csv(file_path, sep='\t', compression='gzip', comment='#')
                else:
                    df = pd.read_csv(file_path, sep='\t', comment='#')
                
                # Extract sample ID from filename
                sample_id = file_path.stem.split('.')[0]
                
                # Use unstranded counts (column 1) or stranded_first (column 2)
                if 'unstranded' in df.columns:
                    counts = df.set_index('gene_id')['unstranded']
                elif len(df.columns) >= 2:
                    counts = df.set_index(df.columns[0])[df.columns[1]]
                else:
                    continue
                
                data_dict[sample_id] = counts
                
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {str(e)}")
                continue
        
        if not data_dict:
            logger.error("No gene expression data loaded")
            return None
        
        # Combine into dataframe
        df = pd.DataFrame(data_dict)
        logger.info(f"Gene expression shape: {df.shape}")
        
        return df
    
    def load_methylation_files(self) -> pd.DataFrame:
        """Load and merge methylation files"""
        
        logger.info("Loading methylation data...")
        methyl_dir = self.input_dir / "methylation"
        
        if not methyl_dir.exists():
            logger.warning("Methylation directory not found")
            return None
        
        files = list(methyl_dir.glob("*.txt*"))
        logger.info(f"Found {len(files)} methylation files")
        
        data_dict = {}
        
        for i, file_path in enumerate(files, 1):
            if i % 50 == 0:
                logger.info(f"Processing file {i}/{len(files)}")
            
            try:
                # Read file
                if file_path.suffix == '.gz':
                    df = pd.read_csv(file_path, sep='\t', compression='gzip')
                else:
                    df = pd.read_csv(file_path, sep='\t')
                
                # Extract sample ID
                sample_id = file_path.stem.split('.')[0]
                
                # Use Beta_value column
                if 'Beta_value' in df.columns:
                    values = df.set_index('Composite Element REF')['Beta_value']
                elif len(df.columns) >= 2:
                    values = df.set_index(df.columns[0])[df.columns[1]]
                else:
                    continue
                
                data_dict[sample_id] = values
                
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {str(e)}")
                continue
        
        if not data_dict:
            logger.error("No methylation data loaded")
            return None
        
        df = pd.DataFrame(data_dict)
        logger.info(f"Methylation shape: {df.shape}")
        
        return df
    
    def load_copy_number_files(self) -> pd.DataFrame:
        """Load and merge copy number files"""
        
        logger.info("Loading copy number data...")
        cnv_dir = self.input_dir / "copy_number"
        
        if not cnv_dir.exists():
            logger.warning("Copy number directory not found")
            return None
        
        files = list(cnv_dir.glob("*.txt*"))
        logger.info(f"Found {len(files)} copy number files")
        
        # CNV data is more complex, we'll create a simplified representation
        data_dict = {}
        
        for i, file_path in enumerate(files, 1):
            if i % 50 == 0:
                logger.info(f"Processing file {i}/{len(files)}")
            
            try:
                df = pd.read_csv(file_path, sep='\t')
                sample_id = file_path.stem.split('.')[0]
                
                # Aggregate CNV data by chromosome
                if 'Chromosome' in df.columns and 'Segment_Mean' in df.columns:
                    cnv_summary = df.groupby('Chromosome')['Segment_Mean'].mean()
                    data_dict[sample_id] = cnv_summary
                
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {str(e)}")
                continue
        
        if not data_dict:
            logger.error("No copy number data loaded")
            return None
        
        df = pd.DataFrame(data_dict)
        logger.info(f"Copy number shape: {df.shape}")
        
        return df
    
    def load_clinical_data(self) -> pd.DataFrame:
        """Load clinical data"""
        
        logger.info("Loading clinical data...")
        clinical_file = self.input_dir / "clinical_data.json"
        
        if not clinical_file.exists():
            logger.warning("Clinical data file not found")
            return None
        
        with open(clinical_file, 'r') as f:
            data = json.load(f)
        
        # Extract relevant clinical information
        clinical_records = []
        
        for case in data['data']['hits']:
            record = {
                'case_id': case['case_id'],
                'submitter_id': case['submitter_id']
            }
            
            # Demographics
            if 'demographic' in case:
                demo = case['demographic']
                record['age'] = demo.get('age_at_index')
                record['gender'] = demo.get('gender')
                record['race'] = demo.get('race')
                record['ethnicity'] = demo.get('ethnicity')
            
            # Diagnosis
            if 'diagnoses' in case and len(case['diagnoses']) > 0:
                diag = case['diagnoses'][0]
                record['primary_diagnosis'] = diag.get('primary_diagnosis')
                record['tumor_stage'] = diag.get('tumor_stage')
                record['vital_status'] = diag.get('vital_status')
                record['days_to_death'] = diag.get('days_to_death')
                record['days_to_last_follow_up'] = diag.get('days_to_last_follow_up')
            
            clinical_records.append(record)
        
        df = pd.DataFrame(clinical_records)
        logger.info(f"Clinical data shape: {df.shape}")
        
        return df
    
    def normalize_data(self, df: pd.DataFrame, method: str = 'robust') -> pd.DataFrame:
        """Normalize omics data"""
        
        logger.info(f"Normalizing data using {method} scaler...")
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            logger.warning(f"Unknown scaling method: {method}")
            return df
        
        # Transpose so samples are rows
        df_t = df.T
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        df_imputed = pd.DataFrame(
            imputer.fit_transform(df_t),
            index=df_t.index,
            columns=df_t.columns
        )
        
        # Scale
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_imputed),
            index=df_imputed.index,
            columns=df_imputed.columns
        )
        
        # Transpose back
        return df_scaled.T
    
    def filter_features(self, df: pd.DataFrame, variance_threshold: float = 0.1,
                       max_features: int = 5000) -> pd.DataFrame:
        """Filter features based on variance"""
        
        logger.info(f"Filtering features (threshold={variance_threshold}, max={max_features})...")
        
        # Calculate variance across samples
        variances = df.var(axis=1)
        
        # Filter by variance threshold
        high_var_features = variances[variances > variance_threshold].index
        logger.info(f"Features with variance > {variance_threshold}: {len(high_var_features)}")
        
        # Select top features by variance
        top_features = variances.nlargest(min(max_features, len(high_var_features))).index
        
        df_filtered = df.loc[top_features]
        logger.info(f"Final features: {df_filtered.shape[0]}")
        
        return df_filtered
    
    def preprocess_pipeline(self, variance_threshold: float = 0.1,
                           max_features: int = 5000):
        """Complete preprocessing pipeline"""
        
        logger.info(f"\n{'='*60}")
        logger.info("Starting preprocessing pipeline")
        logger.info(f"{'='*60}\n")
        
        # Load data
        gene_expr = self.load_gene_expression_files()
        methylation = self.load_methylation_files()
        copy_number = self.load_copy_number_files()
        clinical = self.load_clinical_data()
        
        # Process each omics type
        processed_data = {}
        
        if gene_expr is not None:
            logger.info("\nProcessing gene expression...")
            gene_expr_filtered = self.filter_features(
                gene_expr, variance_threshold, max_features
            )
            gene_expr_norm = self.normalize_data(gene_expr_filtered, 'robust')
            processed_data['gene_expression'] = gene_expr_norm
            
            # Save
            output_file = self.output_dir / "gene_expression_processed.csv"
            gene_expr_norm.to_csv(output_file)
            logger.info(f"Saved to {output_file}")
        
        if methylation is not None:
            logger.info("\nProcessing methylation...")
            methyl_filtered = self.filter_features(
                methylation, variance_threshold, max_features
            )
            methyl_norm = self.normalize_data(methyl_filtered, 'robust')
            processed_data['methylation'] = methyl_norm
            
            # Save
            output_file = self.output_dir / "methylation_processed.csv"
            methyl_norm.to_csv(output_file)
            logger.info(f"Saved to {output_file}")
        
        if copy_number is not None:
            logger.info("\nProcessing copy number...")
            cnv_norm = self.normalize_data(copy_number, 'robust')
            processed_data['copy_number'] = cnv_norm
            
            # Save
            output_file = self.output_dir / "copy_number_processed.csv"
            cnv_norm.to_csv(output_file)
            logger.info(f"Saved to {output_file}")
        
        if clinical is not None:
            logger.info("\nProcessing clinical data...")
            # Save clinical data
            output_file = self.output_dir / "clinical_processed.csv"
            clinical.to_csv(output_file, index=False)
            logger.info(f"Saved to {output_file}")
        
        # Create sample intersection
        logger.info("\nFinding common samples across omics...")
        sample_sets = [df.columns.tolist() for df in processed_data.values()]
        common_samples = set(sample_sets[0])
        for samples in sample_sets[1:]:
            common_samples = common_samples.intersection(set(samples))
        
        logger.info(f"Common samples: {len(common_samples)}")
        
        # Save sample list
        with open(self.output_dir / "common_samples.txt", 'w') as f:
            for sample in sorted(common_samples):
                f.write(f"{sample}\n")
        
        # Save metadata
        metadata = {
            'n_samples': len(common_samples),
            'omics_types': list(processed_data.keys()),
            'shapes': {k: v.shape for k, v in processed_data.items()},
            'variance_threshold': variance_threshold,
            'max_features': max_features
        }
        
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"\n{'='*60}")
        logger.info("Preprocessing complete!")
        logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess TCGA multi-omics data'
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Input directory with raw data'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory for processed files'
    )
    
    parser.add_argument(
        '--variance-threshold',
        type=float,
        default=0.1,
        help='Minimum variance for feature selection'
    )
    
    parser.add_argument(
        '--max-features',
        type=int,
        default=5000,
        help='Maximum number of features per omics'
    )
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = MultiOmicsPreprocessor(args.input_dir, args.output_dir)
    
    # Run preprocessing
    preprocessor.preprocess_pipeline(
        variance_threshold=args.variance_threshold,
        max_features=args.max_features
    )


if __name__ == "__main__":
    main()
