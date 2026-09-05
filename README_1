# Breast Cancer Classification Using Machine Learning

A machine learning project that classifies breast cancer tumors as **Benign (B)** or **Malignant (M)** using the Breast Cancer Wisconsin Diagnostic dataset and Logistic Regression.

> **Disclaimer:** This project is for educational and demonstration purposes only. It is not a medical diagnostic tool and should not be used for clinical decisions.

## 📌 Project Overview

This project demonstrates an end-to-end machine learning workflow:

- Data loading and preprocessing
- Exploratory data analysis
- Feature selection
- Train-test splitting
- Feature scaling using `StandardScaler`
- Binary classification using Logistic Regression
- Model evaluation
- Saving the trained ML pipeline
- Deploying the model through a Streamlit web application

## 🧠 Machine Learning Model

The project uses **Logistic Regression** for binary classification.

| Value | Meaning |
|---|---|
| `0` | Benign |
| `1` | Malignant |

The model uses **30 numerical features** derived from measurements such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

## 📊 Dataset

The project uses the **Breast Cancer Wisconsin Diagnostic dataset**.

- **569 samples**
- **30 numerical input features**
- **357 Benign samples**
- **212 Malignant samples**

The original `id` and empty `Unnamed: 32` columns are removed before training.

## 🔧 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Google Colab / Jupyter Notebook

## ⚙️ Project Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Target Encoding
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
StandardScaler
   ↓
Logistic Regression
   ↓
Model Evaluation
   ↓
Save ML Pipeline
   ↓
Streamlit Web Application
```

## 📈 Model Performance

The model was evaluated on a test set of **114 samples**.

| Metric | Result |
|---|---:|
| Accuracy | 96.49% |
| Precision (Malignant) | ~97.5% |
| Recall (Malignant) | ~92.9% |
| F1 Score (Malignant) | ~95% |
| ROC-AUC | ~99.60% |

### Confusion Matrix

```text
                 Predicted
                B       M
Actual B       71       1
Actual M        3      39
```

These results are based on the project's test split and should not be interpreted as clinical validation.

## 📁 Project Structure

```text
breast-cancer-ml/
│
├── app.py
├── breast_cancer_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

- `app.py` — Streamlit web application
- `breast_cancer_model.pkl` — Saved trained ML pipeline
- `requirements.txt` — Python dependencies
- `README.md` — Project documentation
- `.gitignore` — Files excluded from Git

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd breast-cancer-ml
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

## 💾 Saving the Model

The trained preprocessing and classification pipeline can be saved using Joblib:

```python
import joblib

joblib.dump(pipeline, "breast_cancer_model.pkl")
```

The complete pipeline is saved so that the same feature scaling used during training is applied during prediction.

## 🌐 Deployment

The Streamlit application can be deployed using **Streamlit Community Cloud**.

```text
GitHub Repository
       ↓
Connect Repository
       ↓
Select app.py
       ↓
Install requirements.txt
       ↓
Deploy
       ↓
Web Application
```

## 🔮 Future Improvements

- Compare Logistic Regression with Random Forest, SVM, KNN and Gradient Boosting
- Hyperparameter tuning
- Cross-validation
- Feature selection
- Probability calibration
- Model explainability using SHAP or LIME
- Improved UI/UX
- REST API using FastAPI
- Model monitoring
- Validation on independent datasets

## 👨‍💻 Author

**Piyush Kumar Dash**

B.Tech Computer Science Engineering  
Specialization: Artificial Intelligence & Machine Learning

## ⭐ Acknowledgement

This project was developed as a machine learning learning project to understand the complete workflow from data preprocessing and model training to web application deployment.
