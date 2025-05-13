
---

# Transaction Data Analysis and Insights Project

This project implements an end-to-end pipeline for analyzing financial transaction data, with a focus on data acquisition, processing, exploratory and advanced analysis, anomaly detection, and actionable business insights.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Explanation of Approach](#explanation-of-approach)
- [Guide to Navigating Your Submission](#guide-to-navigating-your-submission)
- [Summary Report](#summary-report)
  - [Key Findings](#key-findings)
  - [Methodology](#methodology)
  - [Recommendations](#recommendations)

---

## Overview

This project provides a comprehensive workflow for transaction data analysis, from data acquisition and cleaning to advanced analytics and business recommendations. The pipeline is designed to handle large-scale datasets efficiently and produce meaningful visualizations and reports for stakeholders.

---

## Project Structure

```
.
├── .env
├── .gitignore
├── .structignore
├── analysis_output/
│   └── visualizations/
├── data_acquisition.log
├── data_processing.ipynb
├── download_checkpoint.json
├── download_data.py
├── level_2.ipynb
├── level_3.ipynb
├── llama_insights_20250513_071823.md
├── output_visualizations/
├── README.md
├── transaction_analysis.py
```

- **Notebooks** (`data_processing.ipynb`, `level_2.ipynb`, `level_3.ipynb`): Data cleaning, advanced analysis, modeling.
- **Scripts** (`download_data.py`, `transaction_analysis.py`): Data acquisition and EDA/visualization.
- **Outputs** (`analysis_output/`, `output_visualizations/`): Visualizations, reports, and flagged transactions.
- **Logs & Configs** (`data_acquisition.log`, `download_checkpoint.json`, `.env`): For reproducibility and troubleshooting.

---

## Setup Instructions

### 1. Prerequisites

- **Python 3.7+**  
- **Google Cloud SDK** (for `gsutil`): Required for data download.
- **Jupyter Notebook** (for `.ipynb` files).
- **pip** or **conda** for package management.

### 2. Clone the Repository

```bash
git clone 
cd 
```

### 3. Environment Setup

(Optional) Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Install dependencies (create `requirements.txt` if not present):

```
pandas
numpy
matplotlib
seaborn
plotly
dask
pyarrow
scikit-learn
tqdm
psutil
swifter
```

```bash
pip install -r requirements.txt
```

### 4. Data Acquisition

- Configure Google Cloud SDK and authenticate.
- Run the data download script:

```bash
python download_data.py
```

- Data will be saved in `transaction_data/` with checkpoints and logs for resuming interrupted downloads.

### 5. Data Processing

- Open `data_processing.ipynb` in Jupyter Notebook.
- Run all cells to clean, transform, and save processed data to `processed_chunks/` in Parquet format.

### 6. Exploratory Data Analysis

- Run the EDA and visualization script:

```bash
python transaction_analysis.py
```

- Visualizations will be saved in `analysis_output/visualizations/`.

### 7. Advanced Analysis & Modeling

- For RFM analysis and anomaly detection, use `level_2.ipynb`.
- For user segmentation and predictive modeling, use `level_3.ipynb`.
- Outputs and reports are saved in `output_visualizations/`.

---

## Explanation of Approach

### 1. Data Acquisition

- Download daily transaction data and reference files from GCP.
- Use checkpointing for robust, resumable downloads.
- Log all download activities and validate data integrity.

### 2. Data Processing

- Read and process raw CSVs in chunks for memory efficiency.
- Clean data (handle missing values, normalize types, deduplicate).
- Engineer features (e.g., time-based, categorical, risk-related).
- Save processed data in Parquet format for fast downstream access.

### 3. Exploratory Data Analysis (EDA)

- Analyze transaction volume, values, and patterns over time.
- Visualize top merchants, categories, and temporal trends.
- Compare online/offline transactions and geographical distributions.

### 4. Advanced Analysis & Modeling

- **RFM Analysis:** Segment users by Recency, Frequency, and Monetary value.
- **Anomaly Detection:** Flag high-risk transactions using rule-based and ML-based methods.
- **User Segmentation:** Cluster users via K-Means or similar algorithms.
- **Prediction:** Build models to forecast user behavior or flag suspicious activity.

---

## Guide to Navigating the Submission

- **`download_data.py`**: Downloads raw data from GCP.
- **`data_processing.ipynb`**: Cleans and processes raw data.
- **`transaction_analysis.py`**: Runs EDA and generates visualizations.
- **`level_2.ipynb`**: Advanced analytics (RFM, anomaly detection).
- **`level_3.ipynb`**: User segmentation and predictive modeling.
- **`analysis_output/visualizations/`**: Basic EDA visualizations.
- **`output_visualizations/`**: Advanced analytics outputs, reports, and flagged transactions.
- **`transaction_analysis_report.md`**: Main summary report of findings.
- **`llama_insights_20250513_071823.md`**: Business insights and recommendations from modeling.
- **`data_acquisition.log`**: Log of data download process.
- **`download_checkpoint.json`**: For resuming downloads.

---

## Summary Report

### Key Findings

- **Temporal Patterns**:  
  - Transaction volume peaks on weekends and during evening hours.
  - Notable seasonal spikes at month-end and during holidays.

- **Merchant and Category Insights**:  
  - Top merchants account for a disproportionate share of transaction value.
  - Certain categories (e.g., electronics, travel) show higher average transaction amounts.

- **Geographical Trends**:  
  - Urban centers dominate transaction counts.
  - Cross-border transactions, though fewer, have higher average values.

- **Risk and Anomaly Detection**:  
  - ~2% of transactions flagged as high-risk, primarily due to unusually high amounts, atypical locations, or high-risk merchant categories.
  - ML-based anomaly detection reveals patterns not captured by rule-based methods, improving fraud detection coverage.

- **User Segmentation**:  
  - Users segmented into distinct clusters (e.g., high-value frequent, low-value infrequent, cross-border shoppers).
  - Each segment exhibits unique behavioral and risk profiles.

### Methodology

1. **Data Acquisition**:  
   - Automated, checkpointed download from GCP ensures completeness and reproducibility.

2. **Data Processing**:  
   - Chunked reading, cleaning, and enrichment for scalability and analytical depth.

3. **Exploratory & Advanced Analysis**:  
   - EDA for trend discovery and hypothesis generation.
   - RFM for customer segmentation.
   - Rule-based and ML-based anomaly detection for fraud/risk insights.

4. **Modeling**:  
   - K-Means clustering for user segments.
   - Predictive models (e.g., Random Forest) for transaction risk scoring.

5. **Reporting & Visualization**:  
   - Interactive HTML and markdown reports for stakeholder communication.
   - CSV exports for further business analysis.

### Recommendations

- **Fraud Monitoring**:  
  - Deploy combined rule-based and ML-based anomaly detection in production for real-time fraud alerts.
  - Periodically review and update risk rules based on emerging patterns.

- **Customer Engagement**:  
  - Target high-value, frequent users with loyalty programs.
  - Develop tailored marketing for distinct user segments identified via clustering.

- **Operational Improvements**:  
  - Focus merchant onboarding efforts on categories and regions with high transaction growth.
  - Monitor cross-border transaction trends for compliance and new business opportunities.

- **Future Enhancements**:  
  - Integrate additional data sources (e.g., device, behavioral) for richer modeling.
  - Experiment with deep learning models for improved anomaly detection.
  - Automate the entire pipeline for regular, scheduled reporting.

---
