# 🏦 AI Credit Scoring System (Credit Risk Assessment UI)

An interactive web application for automated credit risk assessment, built on a Random Forest algorithm. This project demonstrates the full Machine Learning development lifecycle: from data preprocessing and pipeline tuning to AI bias resolution and user interface creation.

---

## ⚠️ Important Disclaimer Regarding Historical Data (AI Bias)

**Model Foundation:** This project uses the classic "German Credit Data" dataset, originally collected in 1994.

During initial analysis and testing, significant logical distortions were identified (Survivorship Bias and historical artifacts). Because of these, **the first version of the application behaved inadequately for modern realities:**
* The model encouraged issuing loans to clients with poor credit histories (in the 90s, having multiple existing loans was often a marker of VIP status).
* The model assigned high scoring points to clients with absolutely no checking account (at the time, this often meant the client held funds in investment accounts or other major banks).

### 🛠 Architectural Solution
To adapt the model to the realities of 2026 and eliminate ethical bias, **the following features were completely removed from the training data:**
1. `credit_history` 
2. `checking_status` 

The current model relies **exclusively on objective mathematics and financial physics**. The primary weights (Feature Importance) are now distributed among:
* Loan Duration (~14%)
* Loan Amount (~12%)
* Borrower's Age (~11%)
* Presence of Savings and Property.

---

## 🖥 User Interface and API (Streamlit)

The application (`app.py`) provides a user-friendly graphical interface designed for bank risk managers.

### How the Interface Works:
1. **Data Input (Frontend):** The user fills out a concise form (Amount, Duration, Age, Purpose, Savings, Property). The biased fields removed from the model are completely hidden from the interface to prevent user confusion.
2. **Processing (Backend):** Data is transformed "on the fly" using a saved category dictionary (One-Hot Encoding) and scaled via `StandardScaler`.
3. **Assessment (Predict Proba):** Instead of a rigid "Yes/No" answer, the model returns the **probability of client reliability** as a percentage, visualized with a progress bar.

### ⚙️ Risk Calibration (Sidebar)
A **"Threshold Calibration"** slider is implemented in the left sidebar.
Because the unbiased model became more conservative (probabilities converged toward the center), the risk manager can dynamically adjust the strictness of the system (e.g., from 0.40 to 0.80). This allows the bank to balance the approval rate and default risk without needing to rewrite any code.

---

## 🛠 Technology Stack

* **Programming Language:** Python 3.10+
* **Machine Learning:** Scikit-Learn (`RandomForestClassifier`, `StandardScaler`, `GridSearchCV`, `Pipeline`)
* **Web Framework:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Serialization:** Joblib

## 🚀 Installation and Run (Local Environment)

**1. Clone the repository:**
```bash
git clone [https://github.com/yourusername/german_credit_scoring.git](https://github.com/yourusername/german_credit_scoring.git)
cd german_credit_scoring
```

**2. Create and activate a virtual environment:**
```bash
python -m venv .venv
# For Windows:
.venv\Scripts\activate
# For macOS/Linux:
source .venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Launch the web application:**
```bash
streamlit run app.py
```
The application will be available in your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```text
german_credit_project/
│
├── data/                    # Raw dataset
│   └── german_credit_data.csv
│
├── models/                  # Saved components (The "Brain" of the app)
│   ├── credit_scoring_model.pkl
│   ├── standard_scaler.pkl
│   └── feature_columns.pkl
│
├── notebooks/               # Jupyter Notebook with analysis (Feature Drops, GridSearch)
│   └── Credit_Scoring_Training.ipynb
│
├── app.py                   # Main Streamlit web application file
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```
