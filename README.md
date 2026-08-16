# BITS Pilani M.Tech (AIML/DSE) — Machine Learning Assignment 2

**Student Name:** Shailendra Kumar  
**Student ID:** 2025ac05816

## a. Problem Statement

Implement multiple classification models on one public classification dataset, evaluate all required models using the specified metrics, and demonstrate the models through an interactive Streamlit web application deployed on Streamlit Community Cloud.

## b. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Source:** UCI Machine Learning Repository  
**Instances:** 569  
**Features:** 30  
**Problem type:** Binary classification  
**Target:** Binary diagnosis label encoded as 0/1

The dataset satisfies the assignment minimum of 500 instances and 12 features.

UCI source:  
https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic

## c. GitHub Repository Link

https://github.com/shailendrakumar5/2025ac05816_ML_Assignment2

## d. Models Used

Exactly the five models explicitly listed in the assignment are implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier — Gaussian
5. Ensemble Model — Random Forest

## Evaluation Metrics

Each model is evaluated using:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

## Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| K-Nearest Neighbors | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9474 | 0.9937 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |

## Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong linear baseline after scaling; best reported Accuracy, AUC and MCC in this run. |
| Decision Tree | Captures non-linear relationships; depth is limited to reduce overfitting. |
| KNN | Distance-based model; scaling is important because feature magnitudes affect distances. |
| Naive Bayes | Fast probabilistic baseline with relatively simple assumptions. |
| Random Forest (Ensemble) | Ensemble of decision trees with strong non-linear classification capability. |

**Overall Winner:** Logistic Regression based on the reported results: Accuracy 0.9825, AUC 0.9954 and MCC 0.9623.

## Repository Structure

```text
project-folder/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── dataset.csv
├── test_data.csv
├── model_metrics.csv
└── models/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── random_forest.pkl
```

## Run Locally

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

Upload `test_data.csv` in the Streamlit application.

## Streamlit Application Requirements

The application includes the assignment-required features:

- CSV test-data upload option
- Model-selection dropdown for the five required models
- Display of Accuracy, AUC, Precision, Recall, F1 Score and MCC
- Confusion matrix
- Classification report
- Results of the different models on the test data

The **Upload test data (CSV)** control and **Select classification model** dropdown are positioned at the top of the Streamlit page for the assignment demonstration.

## Deployment

**GitHub Repository:**  
https://github.com/shailendrakumar5/2025ac05816_ML_Assignment2

**Live Streamlit App:**  
https://2025ac05816mlassignment2-zyvpa4tt96dsyevecpwb9q.streamlit.app

**Streamlit configuration:**
- Branch: `main`
- Main file: `app.py`
- Platform: Streamlit Community Cloud

## BITS Virtual Lab Evidence

The assignment requires one genuine screenshot showing execution on BITS Virtual Lab. The student's own screenshot should be inserted into the final PDF before submission.
