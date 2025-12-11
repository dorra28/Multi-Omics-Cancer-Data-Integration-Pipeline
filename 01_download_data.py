#!/usr/bin/env python3
"""
TCGA Data Download Script
Downloads multi-omics data from TCGA via GDC API

Usage:
    python 01_download_data.py --cancer-type BRCA --output-dir data/raw
"""

import argparse
import json
import os
import requests
import sys
from pathlib import Path
from typing import Dict, List
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TCGADownloader:
    """Downloads TCGA data from GDC API"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # GDC API endpoints
        self.files_endpt = "https://api.gdc.cancer.gov/files"
        self.data_endpt = "https://api.gdc.cancer.gov/data"
        self.cases_endpt = "https://api.gdc.cancer.gov/cases"
        
    def build_query(self, cancer_type: str, data_category: str, 
                   data_type: str = None, workflow_type: str = None) -> Dict:
        """Build GDC API query"""
        
        filters = {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": [f"TCGA-{cancer_type}"]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "data_category",
                        "value": [data_category]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "access",
                        "value": ["open"]
                    }
                }
            ]
        }
        
        if data_type:
            filters["content"].append({
                "op": "in",
                "content": {
                    "field": "data_type",
                    "value": [data_type]
                }
            })
            
        if workflow_type:
            filters["content"].append({
                "op": "in",
                "content": {
                    "field": "analysis.workflow_type",
                    "value": [workflow_type]
                }
            })
        
        params = {
            "filters": json.dumps(filters),
            "fields": "file_id,file_name,cases.submitter_id,data_category,data_type",
            "format": "JSON",
            "size": "1000"
        }
        
        return params
    
    def get_file_ids(self, cancer_type: str, data_category: str, 
                     data_type: str = None, workflow_type: str = None) -> List[Dict]:
        """Get file IDs from GDC"""
        
        params = self.build_query(cancer_type, data_category, data_type, workflow_type)
        
        logger.info(f"Querying GDC for {cancer_type} {data_category} data...")
        response = requests.get(self.files_endpt, params=params)
        
        if response.status_code != 200:
            logger.error(f"API request failed: {response.status_code}")
            return []
        
        data = response.json()
        files = data["data"]["hits"]
        logger.info(f"Found {len(files)} files")
        
        return files
    
    def download_file(self, file_id: str, file_name: str) -> bool:
        """Download single file from GDC"""
        
        output_path = self.output_dir / file_name
        
        if output_path.exists():
            logger.info(f"File already exists: {file_name}")
            return True
        
        logger.info(f"Downloading: {file_name}")
        
        try:
            response = requests.get(
                f"{self.data_endpt}/{file_id}",
                headers={"Content-Type": "application/json"},
                stream=True
            )
            
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"Successfully downloaded: {file_name}")
                return True
            else:
                logger.error(f"Download failed for {file_name}: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading {file_name}: {str(e)}")
            return False
    
    def download_gene_expression(self, cancer_type: str, max_files: int = None):
        """Download gene expression data (RNA-Seq)"""
        
        logger.info(f"\n{'='*60}")
        logger.info("Downloading Gene Expression Data")
        logger.info(f"{'='*60}\n")
        
        files = self.get_file_ids(
            cancer_type=cancer_type,
            data_category="Transcriptome Profiling",
            data_type="Gene Expression Quantification",
            workflow_type="STAR - Counts"
        )
        
        if max_files:
            files = files[:max_files]
        
        gene_expr_dir = self.output_dir / "gene_expression"
        gene_expr_dir.mkdir(exist_ok=True)
        
        success = 0
        for file_info in files:
            file_id = file_info["file_id"]
            file_name = file_info["file_name"]
            
            if self.download_file(file_id, f"gene_expression/{file_name}"):
                success += 1
        
        logger.info(f"\nDownloaded {success}/{len(files)} gene expression files")
    
    def download_methylation(self, cancer_type: str, max_files: int = None):
        """Download DNA methylation data"""
        
        logger.info(f"\n{'='*60}")
        logger.info("Downloading Methylation Data")
        logger.info(f"{'='*60}\n")
        
        files = self.get_file_ids(
            cancer_type=cancer_type,
            data_category="DNA Methylation",
            data_type="Methylation Beta Value"
        )
        
        if max_files:
            files = files[:max_files]
        
        methyl_dir = self.output_dir / "methylation"
        methyl_dir.mkdir(exist_ok=True)
        
        success = 0
        for file_info in files:
            file_id = file_info["file_id"]
            file_name = file_info["file_name"]
            
            if self.download_file(file_id, f"methylation/{file_name}"):
                success += 1
        
        logger.info(f"\nDownloaded {success}/{len(files)} methylation files")
    
    def download_copy_number(self, cancer_type: str, max_files: int = None):
        """Download copy number variation data"""
        
        logger.info(f"\n{'='*60}")
        logger.info("Downloading Copy Number Data")
        logger.info(f"{'='*60}\n")
        
        files = self.get_file_ids(
            cancer_type=cancer_type,
            data_category="Copy Number Variation",
            data_type="Copy Number Segment"
        )
        
        if max_files:
            files = files[:max_files]
        
        cnv_dir = self.output_dir / "copy_number"
        cnv_dir.mkdir(exist_ok=True)
        
        success = 0
        for file_info in files:
            file_id = file_info["file_id"]
            file_name = file_info["file_name"]
            
            if self.download_file(file_id, f"copy_number/{file_name}"):
                success += 1
        
        logger.info(f"\nDownloaded {success}/{len(files)} copy number files")
    
    def download_clinical_data(self, cancer_type: str):
        """Download clinical data"""
        
        logger.info(f"\n{'='*60}")
        logger.info("Downloading Clinical Data")
        logger.info(f"{'='*60}\n")
        
        # Query for clinical data
        filters = {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "project.project_id",
                        "value": [f"TCGA-{cancer_type}"]
                    }
                }
            ]
        }
        
        params = {
            "filters": json.dumps(filters),
            "expand": "diagnoses,demographic,exposures",
            "format": "JSON",
            "size": "10000"
        }
        
        response = requests.get(self.cases_endpt, params=params)
        
        if response.status_code == 200:
            clinical_data = response.json()
            
            output_file = self.output_dir / "clinical_data.json"
            with open(output_file, 'w') as f:
                json.dump(clinical_data, f, indent=2)
            
            logger.info(f"Clinical data saved to {output_file}")
            logger.info(f"Total cases: {len(clinical_data['data']['hits'])}")
        else:
            logger.error(f"Failed to download clinical data: {response.status_code}")
    
    def download_all(self, cancer_type: str, max_files: int = None):
        """Download all omics data types"""
        
        logger.info(f"\n{'#'*60}")
        logger.info(f"# TCGA Data Download: {cancer_type}")
        logger.info(f"# Output directory: {self.output_dir}")
        logger.info(f"{'#'*60}\n")
        
        # Download each data type
        self.download_gene_expression(cancer_type, max_files)
        self.download_methylation(cancer_type, max_files)
        self.download_copy_number(cancer_type, max_files)
        self.download_clinical_data(cancer_type)
        
        logger.info(f"\n{'#'*60}")
        logger.info("# Download Complete!")
        logger.info(f"{'#'*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Download TCGA multi-omics data from GDC'
    )
    
    parser.add_argument(
        '--cancer-type',
        type=str,
        required=True,
        help='TCGA cancer type (e.g., BRCA, LUAD, COAD)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw',
        help='Output directory for downloaded files'
    )
    
    parser.add_argument(
        '--data-types',
        nargs='+',
        choices=['gene_expression', 'methylation', 'copy_number', 'clinical', 'all'],
        default=['all'],
        help='Data types to download'
    )
    
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Maximum number of files to download per data type (for testing)'
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = TCGADownloader(args.output_dir)
    
    # Download data
    if 'all' in args.data_types:
        downloader.download_all(args.cancer_type, args.max_files)
    else:
        if 'gene_expression' in args.data_types:
            downloader.download_gene_expression(args.cancer_type, args.max_files)
        if 'methylation' in args.data_types:
            downloader.download_methylation(args.cancer_type, args.max_files)
        if 'copy_number' in args.data_types:
            downloader.download_copy_number(args.cancer_type, args.max_files)
        if 'clinical' in args.data_types:
            downloader.download_clinical_data(args.cancer_type)


if __name__ == "__main__":
    main()
