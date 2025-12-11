# Multi-Omics Cancer Data Integration Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive bioinformatics pipeline for integrating and analyzing multi-omics cancer data from The Cancer Genome Atlas (TCGA). This pipeline addresses one of the most pressing challenges in modern bioinformatics: effectively combining genomic, transcriptomic, and methylation data to identify cancer subtypes and biomarkers.

## 🎯 Project Overview

This pipeline implements state-of-the-art methods for multi-omics data integration, featuring:
- Automated TCGA data download and preprocessing
- Multi-omics feature extraction using autoencoders
- Late integration for cancer subtype classification
- Survival analysis and biomarker identification
- Comprehensive visualization suite

### Key Features

- **Data Acquisition**: Automated download from TCGA via GDC API
- **Data Processing**: Normalization, quality control, and feature engineering
- **Integration Methods**: Early, middle, and late integration strategies
- **Machine Learning**: Random Forest, XGBoost, and Deep Learning models
- **Interpretability**: SHAP values for biomarker discovery
- **Visualization**: Interactive plots and comprehensive reports

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
# conda (recommended for environment management)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/dorra28/multiomics-cancer-pipeline.git
cd multiomics-cancer-pipeline

# Create conda environment
conda env create -f environment.yml
conda activate multiomics

# Or use pip
pip install -r requirements.txt
```

### Basic Usage

```bash
# 1. Download TCGA data 
python scripts/01_download_data.py --cancer-type BRCA --output-dir data/raw

# 2. Preprocess data
python scripts/02_preprocess.py --input-dir data/raw --output-dir data/processed

# 3. Train integration model
python scripts/03_train_model.py --data-dir data/processed --output-dir models

# 4. Generate predictions and visualizations
python scripts/04_analyze_results.py --model-dir models --output-dir results
```

## 📁 Project Structure

```
multiomics-cancer-pipeline/
│
├── data/
│   ├── raw/                    # Raw TCGA data
│   ├── processed/              # Preprocessed data
│   └── sample_data/            # Small sample dataset for testing
│
├── scripts/
│   ├── 01_download_data.py     # TCGA data download
│   ├── 02_preprocess.py        # Data preprocessing
│   ├── 03_train_model.py       # Model training
│   ├── 04_analyze_results.py   # Results analysis
│   └── utils/
│       ├── data_loader.py
│       ├── preprocessing.py
│       └── visualization.py
│
├── models/
│   ├── autoencoder.py          # Autoencoder for feature extraction
│   ├── integration.py          # Integration models
│   └── classifier.py           # Classification models
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_results_visualization.ipynb
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   └── test_models.py
│
├── results/
│   ├── figures/
│   ├── reports/
│   └── models/
│
├── docs/
│   ├── API.md
│   ├── TUTORIAL.md
│   └── METHODS.md
│
├── requirements.txt
├── environment.yml
├── setup.py
├── README.md
├── LICENSE
└── .gitignore
```

## 📊 Datasets

This pipeline works with TCGA data including:

- **Gene Expression** (RNA-Seq)
- **DNA Methylation** (450K arrays)
- **Copy Number Variation** (CNV)
- **Clinical Data** (survival, demographics)

### Supported Cancer Types

The pipeline supports all 33 TCGA cancer types, including:
- BRCA (Breast Cancer)
- LUAD (Lung Adenocarcinoma)
- COAD (Colon Adenocarcinoma)
- And 30 more...

## 🧠 Methods

### Data Integration Strategies

1. **Early Integration**: Concatenate all omics before analysis
2. **Intermediate Integration**: Learn joint representations
3. **Late Integration**: Combine predictions from individual omics

### Machine Learning Pipeline

```
Raw Data → Preprocessing → Feature Extraction → Integration → Classification → Evaluation
```

#### Feature Extraction
- Autoencoder-based dimensionality reduction
- Variational Autoencoders (VAE) for robust features
- PCA and feature selection

#### Classification
- Random Forest
- XGBoost
- Multi-layer Perceptron (MLP)
- Ensemble methods

#### Evaluation
- Cross-validation (5-fold)
- Performance metrics (Accuracy, F1, AUC-ROC)
- Survival analysis (Kaplan-Meier, Cox regression)
- Biomarker identification (SHAP values)

## 📈 Example Results

```python
from models.integration import MultiOmicsIntegrator
from utils.data_loader import load_tcga_data

# Load data
X_gene, X_methyl, X_cnv, y = load_tcga_data('BRCA')

# Initialize model
model = MultiOmicsIntegrator(
    integration_method='late',
    classifier='xgboost'
)

# Train
model.fit([X_gene, X_methyl, X_cnv], y)

# Predict
predictions = model.predict([X_gene_test, X_methyl_test, X_cnv_test])

# Evaluate
results = model.evaluate(predictions, y_test)
print(f"Accuracy: {results['accuracy']:.3f}")
print(f"F1 Score: {results['f1']:.3f}")
```

## 🔬 Scientific Background

### Why Multi-Omics Integration?

Cancer is a complex disease involving multiple molecular layers. Single-omics analysis provides limited insight:

- **Genomics**: DNA mutations and structural variants
- **Transcriptomics**: Gene expression changes
- **Epigenomics**: DNA methylation patterns
- **Proteomics**: Protein abundance

Integrating these layers reveals:
- More accurate cancer subtypes
- Better prognostic models
- Novel therapeutic targets
- Personalized treatment strategies

### Current Challenges Addressed

1. **High Dimensionality**: 20,000+ features with few samples
2. **Data Heterogeneity**: Different scales, units, and distributions
3. **Missing Data**: Incomplete omics profiles
4. **Interpretability**: Understanding which features drive predictions

## 📚 Documentation

- [Installation Guide](docs/INSTALLATION.md)
- [Tutorial](docs/TUTORIAL.md)
- [API Reference](docs/API.md)
- [Methods Documentation](docs/METHODS.md)
- [Contributing Guidelines](CONTRIBUTING.md)

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_models.py -v

# With coverage
pytest --cov=models --cov-report=html
```

## 📝 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{multiomics_pipeline,
  title = {Multi-Omics Cancer Data Integration Pipeline},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/multiomics-cancer-pipeline}
}
```

### Related Publications

This pipeline implements methods from:
- The Cancer Genome Atlas Research Network (2012)
- Subramanian et al. (2020) - Multi-omics Data Integration
- Chaudhary et al. (2018) - Deep Learning for Survival Prediction

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The Cancer Genome Atlas (TCGA) for providing comprehensive cancer datasets
- Genomic Data Commons (GDC) for data hosting and API access
- The bioinformatics community for open-source tools and methods

## 📧 Contact

- **Author**: Dorra Rjaibi
- **Email**: dorra.rjaibi@pasteur.com
- **Issues**: [GitHub Issues]([https://github.com/dorra28/Multi-Omics-Cancer-Data-Integration-Pipeline/) ] 

## 🔗 Useful Links

- [TCGA Data Portal](https://portal.gdc.cancer.gov/)
- [GDC API Documentation](https://docs.gdc.cancer.gov/API/Users_Guide/Getting_Started/)
- [UCSC Xena Browser](https://xenabrowser.net/)
- [cBioPortal](https://www.cbioportal.org/)

---

**Note**: This pipeline is for research purposes only. Not intended for clinical use.
