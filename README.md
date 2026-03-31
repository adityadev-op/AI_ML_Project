# High-value Customers Identification for E-Commerce

This project implements a BYOP capstone on identifying high-value customers for an e-commerce company using synthetic data, feature engineering, and machine learning.

## What is included
- `src/high_value_customers.py`: Data generation, feature engineering, modeling pipeline.
- `src/run.py`: Orchestrator to reproduce the full workflow.
- `requirements.txt`: Dependency list.
- `project_report.md`: Structured project report.

## Setup
1. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate  # windows
source venv/bin/activate  # unix
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python src/run.py
```

This will generate `data/customers.csv`, train models, evaluate performance, and create `models/high_value_customer_model.joblib`.

## Usage
- Input: synthetic customer data.
- Output: classification report and feature importance.
- High-value label is top 20% by revenue with a target classification task.

## Notes
- Replace synthetic data generator with real e-commerce dataset using the same feature schema.
- Use this as a template for end-to-end modeling and submission.
