# readmission-prediction

# Hospital Readmission Prediction Project

## 🔍 Overview

This project explores binary classification to predict whether a patient will be readmitted to hospital within 30 days. It includes feature engineering, model training with Random Forest, and evaluation using ROC AUC and F1-score. It also outlines an approach for Named Entity Recognition (NER) using clinical discharge notes.

## 📁 Files

* `hospital_readmission_analysis.py`: Main script for data processing, model training, and evaluation
* `final_detailed_readmission_report.pdf`: Concise project summary and discussion
* `requirements.txt`: Dependencies required to run the project

## 📊 Instructions

1. **Clone repository:**

   ```bash
   git clone https://github.com/12sf12/readmission-prediction
   cd readmission-prediction
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analysis:**

   ```bash
   python hospital_readmission_analysis.py
   ```

## 🧠 Optional (NER with Clinical Notes)

For clinical Named Entity Recognition (NER), install:

```bash
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_sm-0.5.0.tar.gz
```

And modify the script to load the `en_core_sci_sm` model to extract diagnoses, medications, etc.

## 📌 Notes

* Model performance is limited due to small dataset size (200 rows)
* Discharge note NER was not executed in-code due to environment limitations

## 🧪 Tested On

* Python 3.11+
* scikit-learn 1.2+
* pandas 1.4+
* matplotlib/seaborn for visualizations

## 📄 License

MIT License
