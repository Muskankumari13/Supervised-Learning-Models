# Titanic Survival Prediction

This repository presents a supervised machine learning pipeline to predict passenger survival on the Titanic dataset.  
The dataset originates from the well-known Kaggle Titanic competition and serves as a benchmark problem for classification tasks.

---

## Project Overview
The objective of this project is to build, train, and evaluate multiple supervised classification models to determine survival outcomes based on passenger attributes such as:

- Passenger Class (Pclass)  
- Gender (Sex)  
- Age  
- Fare  
- Embarked Port  
- Family-related features (SibSp, Parch)  

---

## Methodology
1. **Data Preprocessing**
   - Handling missing values (Age, Embarked, Cabin).  
   - Encoding categorical variables using label encoding and one-hot encoding.  
   - Outlier detection and removal (IQR method applied to Age).  

2. **Feature Engineering**
   - Selection of relevant features.  
   - Transformation of categorical features.  
   - Optional scaling of continuous features.  

3. **Model Development**
   The following supervised learning models were implemented and trained:
   - Logistic Regression  
   - K-Nearest Neighbors (KNN)  
   - Support Vector Machine (SVM)  
   - Decision Tree Classifier  
   - Random Forest Classifier  
   - Gradient Boosting Classifier  
   - XGBoost Classifier  
   - LightGBM Classifier  

4. **Model Evaluation**
   - Accuracy score comparison.  
   - Visualization of model performance.  
   - Ranking models by predictive accuracy.  

---

## Results
| Model                                | Accuracy |
|--------------------------------------|----------|
| Decision Tree Classifier             | 0.826389 |
| LightGBM (Light Gradient Boosting)   | 0.819444 |
| XGBoost (Extreme Gradient Boosting)  | 0.819444 |
| Gradient Boosting Classifier         | 0.812500 |
| Support Vector Machine (SVM)         | 0.805556 |
| Random Forest Classifier             | 0.805556 |
| KNN (K-Nearest Neighbor)             | 0.798611 |
| Logistic Regression                  | 0.777778 |

The **Decision Tree Classifier** achieved the highest accuracy among the tested models, closely followed by LightGBM and XGBoost.

---

## Technology Stack
- **Programming Language:** Python  
- **Data Manipulation:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Machine Learning Frameworks:** Scikit-learn, XGBoost, LightGBM  

---

## How to Reproduce
1. Clone the repository:
   ```bash
   git clone https:https://github.com/Muskankumari13/Supervised-Learning-Models
