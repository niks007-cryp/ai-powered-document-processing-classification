# AI-Powered Document Processing & Classification

An API-key-free NLP pipeline that classifies incoming business documents and extracts a small set of operational fields so documents can be routed into the correct workflow automatically.

## Problem
Manual document triage is repetitive and error-prone. A useful automation layer should first identify the document type and then return structured fields for downstream processing.

## What it does
- TF-IDF text representation with word n-grams
- Logistic Regression multi-class classifier
- Four document classes: invoice, loan application, employment letter, complaint
- Stratified holdout evaluation
- Regex-based field extraction for document ID, customer, amount and date
- Fully local execution; no API keys or external model calls
- Automated tests

## Run
```bash
pip install -r requirements.txt
PYTHONPATH=src python -c "from pipeline import train; print(train()[1])"
pytest -q
```

## Design
`document text -> TF-IDF -> classifier -> document type -> field extraction -> structured record`

The classifier and deterministic extraction stages are intentionally separate: classification can be upgraded without changing the downstream extraction contract.

## Verified result
With seed `42`, 480 generated documents and a 25% stratified holdout, the classifier achieved **100% accuracy and 100% macro-F1** in the verified run. The test suite passed **3/3 tests**.

The perfect score is expected for this controlled synthetic benchmark and should not be presented as evidence of production-grade document understanding. Real documents require OCR/layout handling, noisy text evaluation, entity validation and human-review fallbacks.
