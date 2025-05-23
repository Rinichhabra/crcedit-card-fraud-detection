# credit-card-fraud-detection
Machine learning project for detecting fraudulent credit card transactions using multiple classification models.
# Credit Card Fraud Detection

## Project Overview

This project involves detecting fraudulent credit card transactions using machine learning techniques. The dataset contains anonymized credit card transaction data, with features representing transaction details and a binary target variable indicating fraud.

The goal is to build classification models that can accurately identify fraudulent transactions and minimize false positives and false negatives.

---

## Dataset

- **creditcard.csv**: The dataset includes transaction records with features such as transaction amount, time, and anonymized principal components.
- The dataset is highly imbalanced, with very few fraud cases compared to non-fraud cases.

---

## Files in this Repository

- `creditcardfraud_detection.ipynb`: Jupyter Notebook containing the complete exploratory data analysis (EDA), data preprocessing, modeling, and evaluation.
- `Credit Card Fraud Detection final.pptx`: Presentation summarizing the project, methodology, and key findings.
- `README.md`: This file.
- Various PNG images for visualizations such as correlation heatmaps, histograms, and boxplots.
- `.gitignore`: To exclude large files or unnecessary files from Git tracking.

---

## Key Visualizations

- Correlation heatmap of features
- Histograms of selected features
- Boxplot of transaction amounts by class (fraud or non-fraud)
- Distribution of classes (fraud vs. non-fraud)

---

## Machine Learning Models Used

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Naive Bayes
- XGBoost (optional)

Each model was evaluated using accuracy, precision, recall, F1-score, and ROC-AUC curves.

---

## How to Use

1. Clone this repository.
2. Open the Jupyter Notebook `creditcardfraud_detection.ipynb`.
3. Follow through the notebook to reproduce the analysis and results.
4. Refer to the presentation `Credit Card Fraud Detection final.pptx` for a project summary.

---

## Note on Dataset Size

The original `creditcard.csv` dataset is quite large (~144 MB). Due to GitHub file size limits, the dataset may be excluded or managed using Git Large File Storage (Git LFS). Please download the dataset separately if necessary.

---

## Contact

For any questions or feedback, please contact:

**Rini Chhabra**  
Email: rinisamuel27@gmail.com  
GitHub: [Rinichhabra](https://github.com/Rinichhabra)

---

*Thank you for checking out this project!*
