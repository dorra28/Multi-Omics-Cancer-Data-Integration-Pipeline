# Project Structure & Quick Reference

##  Complete Directory Structure

```
multiomics-cancer-pipeline/
│
├──  README.md                          # Project overview and quick start
├──  LICENSE                            # MIT License
├──  requirements.txt                   # Python dependencies
├──  environment.yml                    # Conda environment
├──  setup.py                           # Package installation
├──  .gitignore                         # Git ignore rules
│
├──  scripts/                           # Main pipeline scripts
│   ├── 01_download_data.py              # TCGA data download
│   ├── 02_preprocess.py                 # Data preprocessing
│   ├── 03_train_model.py                # Model training
│   ├── 04_analyze_results.py            # Results analysis
│   └── utils/                           # Utility modules
│       ├── __init__.py
│       ├── data_loader.py               # Data loading utilities
│       ├── preprocessing.py             # Preprocessing functions
│       ├── visualization.py             # Plotting functions
│       └── metrics.py                   # Evaluation metrics
│
├──  models/                            # Model architectures
│   ├── __init__.py
│   ├── autoencoder.py                   # Autoencoder models
│   ├── integration.py                   # Integration strategies
│   └── classifier.py                    # Classification models
│
├──  data/                              # Data directory (not in git)
│   ├── raw/                             # Raw downloaded data
│   │   └── [CANCER_TYPE]/
│   │       ├── gene_expression/
│   │       ├── methylation/
│   │       ├── copy_number/
│   │       └── clinical_data.json
│   ├── processed/                       # Processed data
│   │   └── [CANCER_TYPE]/
│   │       ├── gene_expression_processed.csv
│   │       ├── methylation_processed.csv
│   │       ├── copy_number_processed.csv
│   │       ├── clinical_processed.csv
│   │       ├── common_samples.txt
│   │       └── metadata.json
│   └── sample_data/                     # Sample data for testing
│       └── processed/
│
├── results/                           # Analysis results (not in git)
│   └── [CANCER_TYPE]/
│       ├── figures/                     # Generated plots
│       │   ├── confusion_matrix.png
│       │   ├── roc_curves.png
│       │   └── feature_importance.png
│       └── reports/                     # HTML reports
│           └── analysis_report.html
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_exploratory_analysis.ipynb   # Data exploration
│   ├── 02_model_training.ipynb         # Interactive training
│   ├── 03_results_visualization.ipynb  # Results visualization
│   └── 04_survival_analysis.ipynb      # Survival analysis
│
├── tests/                            # Unit tests
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_integration.py
│
├── docs/                             # Documentation
│   ├── API.md                          # API documentation
│   ├── TUTORIAL.md                     # Step-by-step tutorial
│   ├── METHODS.md                      # Method descriptions
│   ├── INSTALLATION.md                 # Installation guide
│   ├── CANCER_TYPES.md                 # Supported cancer types
│   └── CONTRIBUTING.md                 # Contribution guidelines
│
└── config/                           # Configuration files
    ├── default_config.yaml             # Default parameters
    └── cancer_configs/                 # Cancer-specific configs
        ├── BRCA.yaml
        ├── LUAD.yaml
        └── COAD.yaml
```

##  Quick Commands Reference

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/multiomics-cancer-pipeline.git
cd multiomics-cancer-pipeline

# Install with conda
conda env create -f environment.yml
conda activate multiomics

# Or with pip
pip install -r requirements.txt
```

### Complete Pipeline (One Cancer Type)

```bash
# Step 1: Download (example: Breast Cancer)
python scripts/01_download_data.py \
    --cancer-type BRCA \
    --output-dir data/raw/BRCA \
    --max-files 100  # Optional: limit for testing

# Step 2: Preprocess
python scripts/02_preprocess.py \
    --input-dir data/raw/BRCA \
    --output-dir data/processed/BRCA \
    --variance-threshold 0.1 \
    --max-features 5000

# Step 3: Train Model
python scripts/03_train_model.py \
    --data-dir data/processed/BRCA \
    --output-dir models/BRCA \
    --integration-method early \
    --n-components 50 \
    --label-type tumor_stage

# Step 4: Analyze Results
python scripts/04_analyze_results.py \
    --model-dir models/BRCA \
    --output-dir results/BRCA
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_preprocessing.py -v

# With coverage report
pytest tests/ --cov=scripts --cov=models --cov-report=html
```

### Jupyter Notebooks

```bash
# Start Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook
```

## Data Flow Diagram

```
┌─────────────────┐
│   TCGA/GDC      │
│   Data Portal   │
└────────┬────────┘
         │
         │ 01_download_data.py
         ▼
┌─────────────────┐
│   Raw Data      │
│ - Gene Expr     │
│ - Methylation   │
│ - Copy Number   │
│ - Clinical      │
└────────┬────────┘
         │
         │ 02_preprocess.py
         ▼
┌─────────────────┐
│ Processed Data  │
│ - Normalized    │
│ - Filtered      │
│ - Aligned       │
└────────┬────────┘
         │
         │ 03_train_model.py
         ▼
┌─────────────────┐
│ Trained Models  │
│ - Integration   │
│ - Classifiers   │
└────────┬────────┘
         │
         │ 04_analyze_results.py
         ▼
┌─────────────────┐
│    Results      │
│ - Metrics       │
│ - Visualizations│
│ - Reports       │
└─────────────────┘
```

## Configuration

### Environment Variables

Create a `.env` file:

```bash
# Optional: GDC authentication token for controlled data
GDC_TOKEN=your_token_here

# Computational settings
N_JOBS=-1  # Use all CPU cores
MEMORY_LIMIT=16GB
```

### Config Files

Create `config/custom_config.yaml`:

```yaml
# Data Download
download:
  max_files: 100
  data_types: [gene_expression, methylation, copy_number, clinical]

# Preprocessing
preprocessing:
  variance_threshold: 0.1
  max_features: 5000
  normalization_method: robust
  imputation_strategy: median

# Model Training
training:
  integration_method: early
  n_components: 50
  label_type: tumor_stage
  test_size: 0.2
  cv_folds: 5
  
  # Model parameters
  random_forest:
    n_estimators: 200
    max_depth: 15
    min_samples_split: 10
```

##  Common Workflows

### Workflow 1: Single Cancer Type Analysis

```bash
#!/bin/bash
# run_single_cancer.sh

CANCER_TYPE="BRCA"

python scripts/01_download_data.py --cancer-type $CANCER_TYPE --output-dir data/raw/$CANCER_TYPE
python scripts/02_preprocess.py --input-dir data/raw/$CANCER_TYPE --output-dir data/processed/$CANCER_TYPE
python scripts/03_train_model.py --data-dir data/processed/$CANCER_TYPE --output-dir models/$CANCER_TYPE
python scripts/04_analyze_results.py --model-dir models/$CANCER_TYPE --output-dir results/$CANCER_TYPE

echo "Analysis complete for $CANCER_TYPE"
```

### Workflow 2: Multi-Cancer Comparison

```bash
#!/bin/bash
# run_multi_cancer.sh

CANCER_TYPES=("BRCA" "LUAD" "COAD" "PRAD")

for CANCER in "${CANCER_TYPES[@]}"; do
    echo "Processing $CANCER..."
    
    python scripts/01_download_data.py --cancer-type $CANCER --output-dir data/raw/$CANCER --max-files 50
    python scripts/02_preprocess.py --input-dir data/raw/$CANCER --output-dir data/processed/$CANCER
    python scripts/03_train_model.py --data-dir data/processed/$CANCER --output-dir models/$CANCER
    python scripts/04_analyze_results.py --model-dir models/$CANCER --output-dir results/$CANCER
    
    echo "Completed $CANCER"
done

echo "All analyses complete"
```

### Workflow 3: Hyperparameter Tuning

```python
# scripts/tune_hyperparameters.py

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")
```

##  Key Files Explained

### 01_download_data.py
- Downloads multi-omics data from TCGA via GDC API
- Supports selective download (specific data types)
- Handles authentication for controlled data
- Saves metadata about downloads

### 02_preprocess.py
- Loads raw omics files
- Performs quality control
- Normalizes and scales data
- Filters low-variance features
- Finds common samples across omics
- Saves processed data matrices

### 03_train_model.py
- Loads processed data
- Applies dimensionality reduction (PCA)
- Implements integration strategies
- Trains classification models
- Performs cross-validation
- Saves trained models

### 04_analyze_results.py
- Loads trained models
- Generates performance metrics
- Creates visualizations
- Produces HTML reports
- Performs biomarker analysis

##  Example Use Cases

### Use Case 1: Cancer Subtype Discovery

```python
# Discover breast cancer subtypes
python scripts/03_train_model.py \
    --data-dir data/processed/BRCA \
    --output-dir models/BRCA_subtypes \
    --integration-method early \
    --label-type primary_diagnosis
```

### Use Case 2: Survival Prediction

```python
# Predict patient survival
python scripts/03_train_model.py \
    --data-dir data/processed/BRCA \
    --output-dir models/BRCA_survival \
    --integration-method late \
    --label-type vital_status
```

### Use Case 3: Treatment Response

```python
# Predict treatment response
python scripts/03_train_model.py \
    --data-dir data/processed/LUAD \
    --output-dir models/LUAD_response \
    --integration-method early \
    --label-type treatment_response
```

##  Additional Resources

### Documentation
- [Full API Docs](docs/API.md)
- [Detailed Tutorial](docs/TUTORIAL.md)
- [Method Details](docs/METHODS.md)

### External Links
- [TCGA Website](https://www.cancer.gov/tcga)
- [GDC Data Portal](https://portal.gdc.cancer.gov/)
- [Paper References](docs/REFERENCES.md)

### Community
- [GitHub Issues](https://github.com/dorra28/multiomics-cancer-pipeline/issues)
- [Discussions](https://github.com/dorra28/multiomics-cancer-pipeline/discussions)

## Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| Download timeout | Use `--max-files` to limit downloads |
| Out of memory | Reduce `--max-features` in preprocessing |
| Low accuracy | Check class balance, increase samples |
| Missing samples | Normal for TCGA, need intersection |
| Import errors | Verify `pip install -r requirements.txt` |

## Expected Results

### Typical Performance (BRCA, tumor stage)
- Training Accuracy: 75-85%
- Test Accuracy: 65-75%
- CV Score: 70-80%

### Processing Times (1000 samples)
- Download: 30-60 min
- Preprocessing: 5-10 min
- Training: 10-15 min
- Analysis: 2-5 min

## Learning Path

1. **Beginner**: Run with sample data
2. **Intermediate**: Download and process one cancer type
3. **Advanced**: Implement custom integration methods
4. **Expert**: Develop new models and publish results

## License & Citation

MIT License 

If using in research:
```bibtex
@software{multiomics_pipeline,
  title = {Multi-Omics Cancer Data Integration Pipeline},
  author = {Dorra Rjaibi},
  year = {2025},
  url = {https://github.com/dorra28/multiomics-cancer-pipeline}
}
```
