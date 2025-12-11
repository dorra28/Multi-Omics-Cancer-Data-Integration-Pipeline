# Multi-Omics Integration Pipeline - Tutorial

This tutorial will guide you through using the multi-omics integration pipeline step by step.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Detailed Workflow](#detailed-workflow)
4. [Advanced Usage](#advanced-usage)
5. [Troubleshooting](#troubleshooting)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/multiomics-cancer-pipeline.git
cd multiomics-cancer-pipeline
```

### Step 2: Create Environment

Using conda (recommended):

```bash
conda env create -f environment.yml
conda activate multiomics
```

Using pip:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import pandas, sklearn, numpy; print('Installation successful!')"
```

## Quick Start

### Running with Sample Data

We provide a small sample dataset for testing:

```bash
# The pipeline includes sample data in data/sample_data/
python scripts/03_train_model.py \
    --data-dir data/sample_data/processed \
    --output-dir results/sample_run
```

### Running with TCGA Data

```bash
# 1. Download data (Breast Cancer example)
python scripts/01_download_data.py \
    --cancer-type BRCA \
    --output-dir data/raw/BRCA \
    --max-files 50  # Limit for testing

# 2. Preprocess
python scripts/02_preprocess.py \
    --input-dir data/raw/BRCA \
    --output-dir data/processed/BRCA

# 3. Train model
python scripts/03_train_model.py \
    --data-dir data/processed/BRCA \
    --output-dir models/BRCA

# 4. Analyze results
python scripts/04_analyze_results.py \
    --model-dir models/BRCA \
    --output-dir results/BRCA
```

## Detailed Workflow

### 1. Data Download

The download script fetches multi-omics data from TCGA via the GDC API.

#### Supported Cancer Types

All 33 TCGA cancer types are supported. Common examples:

| Code | Cancer Type |
|------|-------------|
| BRCA | Breast Invasive Carcinoma |
| LUAD | Lung Adenocarcinoma |
| COAD | Colon Adenocarcinoma |
| PRAD | Prostate Adenocarcinoma |
| KIRC | Kidney Renal Clear Cell Carcinoma |

[Full list of cancer types](https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tcga-study-abbreviations)

#### Download Specific Data Types

```bash
# Only gene expression
python scripts/01_download_data.py \
    --cancer-type BRCA \
    --data-types gene_expression \
    --output-dir data/raw/BRCA

# Gene expression and methylation
python scripts/01_download_data.py \
    --cancer-type BRCA \
    --data-types gene_expression methylation \
    --output-dir data/raw/BRCA

# All data types
python scripts/01_download_data.py \
    --cancer-type BRCA \
    --data-types all \
    --output-dir data/raw/BRCA
```

#### Download Options

- `--max-files N`: Limit files per data type (useful for testing)
- `--data-types`: Choose which omics to download

### 2. Data Preprocessing

The preprocessing script:
- Loads raw omics files
- Filters low-variance features
- Normalizes data
- Handles missing values
- Finds common samples across omics

#### Preprocessing Options

```bash
python scripts/02_preprocess.py \
    --input-dir data/raw/BRCA \
    --output-dir data/processed/BRCA \
    --variance-threshold 0.1 \    # Keep features with variance > 0.1
    --max-features 5000           # Top 5000 features per omics
```

#### Output Files

```
data/processed/BRCA/
├── gene_expression_processed.csv    # Normalized gene expression
├── methylation_processed.csv        # Normalized methylation
├── copy_number_processed.csv        # Normalized CNV
├── clinical_processed.csv           # Clinical data
├── common_samples.txt               # Sample IDs across all omics
└── metadata.json                    # Processing metadata
```

### 3. Model Training

The training script implements multiple integration strategies:

#### Early Integration

Concatenates all omics before training:

```bash
python scripts/03_train_model.py \
    --data-dir data/processed/BRCA \
    --output-dir models/BRCA_early \
    --integration-method early \
    --n-components 50 \
    --label-type tumor_stage
```

#### Late Integration

Trains separate models per omics:

```bash
python scripts/03_train_model.py \
    --data-dir data/processed/BRCA \
    --output-dir models/BRCA_late \
    --integration-method late \
    --n-components 50 \
    --label-type tumor_stage
```

#### Label Options

Choose what to predict:

- `tumor_stage`: Cancer stage (I, II, III, IV)
- `primary_diagnosis`: Diagnosis type
- `vital_status`: Alive/Dead
- `gender`: Male/Female

#### Output Files

```
models/BRCA/
├── early_integration_model.pkl      # Trained model
├── confusion_matrix.png             # Performance visualization
├── feature_importance.png           # Top features
└── training_summary.json            # Training metrics
```

### 4. Results Analysis

Analyze trained models and generate reports:

```bash
python scripts/04_analyze_results.py \
    --model-dir models/BRCA \
    --data-dir data/processed/BRCA \
    --output-dir results/BRCA
```

This generates:
- Performance metrics
- ROC curves
- Survival analysis (if applicable)
- Biomarker identification
- HTML report

## Advanced Usage

### Custom Label Creation

Create custom labels from clinical data:

```python
import pandas as pd

# Load clinical data
clinical = pd.read_csv('data/processed/BRCA/clinical_processed.csv')

# Create custom label (e.g., high-risk vs low-risk)
def create_risk_label(row):
    if row['tumor_stage'] in ['stage iv', 'stage iiic']:
        return 'high_risk'
    else:
        return 'low_risk'

clinical['risk_group'] = clinical.apply(create_risk_label, axis=1)
clinical.to_csv('data/processed/BRCA/clinical_processed.csv', index=False)
```

### Feature Selection

Customize feature selection:

```python
from sklearn.feature_selection import SelectKBest, f_classif

# In preprocessing script, add:
selector = SelectKBest(f_classif, k=1000)
X_selected = selector.fit_transform(X, y)
```

### Hyperparameter Tuning

Optimize model parameters:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

### Deep Learning Integration

Use autoencoders for feature extraction:

```python
import torch
import torch.nn as nn

class OmicsAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.Linear(1000, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
```

### Survival Analysis

Perform survival analysis:

```python
from lifelines import KaplanMeierFitter, CoxPHFitter

# Kaplan-Meier
kmf = KaplanMeierFitter()
kmf.fit(
    durations=clinical['days_to_death'],
    event_observed=clinical['vital_status'] == 'Dead'
)
kmf.plot()

# Cox Proportional Hazards
cph = CoxPHFitter()
cph.fit(
    clinical[['age', 'tumor_stage', 'days_to_death', 'vital_status']],
    duration_col='days_to_death',
    event_col='vital_status'
)
cph.print_summary()
```

## Troubleshooting

### Common Issues

#### 1. Download Fails

**Problem**: API requests timing out

**Solution**:
```bash
# Use --max-files to download in smaller batches
python scripts/01_download_data.py \
    --cancer-type BRCA \
    --max-files 10 \
    --output-dir data/raw/BRCA
```

#### 2. Memory Issues

**Problem**: Out of memory during preprocessing

**Solution**:
```bash
# Reduce max features
python scripts/02_preprocess.py \
    --input-dir data/raw/BRCA \
    --output-dir data/processed/BRCA \
    --max-features 1000
```

#### 3. No Common Samples

**Problem**: Different samples in each omics

**Solution**: This is normal for TCGA data. The pipeline automatically finds the intersection of samples. You need at least 30 common samples for meaningful analysis.

#### 4. Low Model Performance

**Problem**: Poor classification accuracy

**Solutions**:
- Check label distribution (need balanced classes)
- Increase number of features
- Try different label types
- Ensure sufficient samples per class (min 20)

### Getting Help

1. Check the [API documentation](docs/API.md)
2. Review [method details](docs/METHODS.md)
3. Open an [issue on GitHub](https://github.com/yourusername/multiomics-cancer-pipeline/issues)
4. Contact: your.email@example.com

## Best Practices

1. **Start Small**: Test with `--max-files 50` before full download
2. **Document Labels**: Keep track of which clinical variables you use
3. **Version Control**: Commit preprocessing parameters to git
4. **Save Results**: Store training summaries for comparison
5. **Validate Findings**: Cross-validate with literature

## Next Steps

- Explore [Jupyter notebooks](../notebooks/) for interactive analysis
- Read about [integration methods](docs/METHODS.md)
- Try [different cancer types](docs/CANCER_TYPES.md)
- Contribute improvements via pull requests

## Citations

If you use this pipeline, please cite:

```bibtex
@software{multiomics_pipeline,
  title = {Multi-Omics Cancer Data Integration Pipeline},
  author = {Rjaibi Dorra},
  year = {2025},
  url = {https://github.com/dorra28/multiomics-cancer-pipeline}
}
```
