# SQL Injection Detection System using Machine Learning

This project implements a system that leverages Machine Learning algorithms (Random Forest, Naive Bayes, and XGBoost) to detect potential SQL Injection (SQLi) attacks from SQL queries. The project includes a comprehensive workflow for data preparation from multiple sources, model training, evaluation on a common test set, and a web application demo (Flask) for real-time detection.

**Course Information:**
* **University:** University of Information Technology - VNUHCM
* **Course:** Introduction to Information Assurance and Security - IE105.P21
* **Instructor:** Dr. Nguyen Tan Cam
* **Students:**
    * Nguyen Dang Quang Phuc - 23521204
    * Tran Thi Nhu Phuong - 23521249

---

## 🚀 Key Features

* **Multi-Source Data Handling:** Ingests and standardizes data from multiple source files with varying formats (delimiters, headers).
* **Robust Training & Evaluation:** Trains multiple ML models (Random Forest, Naive Bayes, XGBoost) on different training sets and evaluates them against a unified, common test set.
* **Automated Reporting:** Generates performance comparison reports (Accuracy, Precision, Recall, F1-Score) for all trained models.
* **Interactive Web Demo:** A Flask-based web application to test the best-performing model with user-provided SQL queries.

---

## 📋 Project Workflow

1.  **Source Data Preparation:** Place four source dataset files (e.g., `dataset1.csv`, `dataset2.csv`, etc.) into the `data/` directory.
2.  **Data Standardization & Splitting (`prepare_common_test_set.py`):**
    * This script reads the four source files, handling their different formats.
    * It extracts 20% from each source file to create a single, common test set named `datatest.csv`.
    * The remaining 80% from each source file is saved as a standardized training file (e.g., `dataset1_train_std.csv`).
    * **Standardized Format:** All generated files are headerless and use a semicolon (`;`) as the delimiter.
3.  **Model Training & Evaluation (`training.py`):**
    * This script trains the ML models on each `*_train_std.csv` file.
    * All trained models are then evaluated against the common `datatest.csv`.
    * Detailed evaluation results are saved to `model/evaluation_results_on_common_test.json`.
    * Trained models and vectorizers are saved to the `model/` directory.
4.  **Performance Reporting (`evaluation_reporter.py`):**
    * Reads the results JSON file and generates performance comparison tables for all models.
5.  **Web Application Demo (`app.py`):**
    * Loads the best-performing model and vectorizer for real-time SQLi detection via a web interface.

---

## 🛠️ Installation

1.  **Prerequisites:**
    * Python 3.8+
    * pip (Python package manager)
    * Homebrew (for macOS users, to install `libomp` for XGBoost if needed)

2.  **Create and Activate a Virtual Environment:**
    From the project's root directory:
    ```bash
    python3 -m venv venv
    ```
    Activate the environment:
    * On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
    * On Windows:
        ```bash
        venv\Scripts\activate
        ```

3.  **Install `libomp` (macOS only, if XGBoost fails):**
    ```bash
    brew install libomp
    ```

4.  **Install Required Python Libraries:**
    (Ensure the virtual environment is activated)
    ```bash
    pip3 install -r requirements.txt
    ```

---

## ⚙️ How to Run

1.  **Prepare Source Data:**
    * Place your four source dataset files in the `data/` directory.
    * **Crucially:** Open `prepare_common_test_set.py` and update the `SOURCE_DATASET_FILES_CONFIG` dictionary to match the filenames and properties (delimiter, header, encoding) of your source files.

2.  **Run the Data Preparation Script (Run once, or when source data changes):**
    ```bash
    python3 prepare_common_test_set.py
    ```
    This will generate the standardized `*_train_std.csv` and `datatest.csv` files in the `data/` directory.

3.  **Train the Models:**
    ```bash
    python3 training.py
    ```

4.  **View the Performance Report:**
    ```bash
    python3 evaluation_reporter.py
    ```
    Use this report to determine the best-performing model for the web app.

5.  **Run the Web Application Demo:**
    * Open `app.py` and update the `DATASET_FOR_WEB` variable to the name of the training set that produced the best model (e.g., `DATASET_FOR_WEB = "dataset4_train_std"`).
    * Run the app:
        ```bash
        python3 app.py
        ```
    * Open your browser and navigate to `http://127.0.0.1:5000`.

---

## 📁 Project Structure

```
SQL_INJECTION_DETECTOR/
├── data/
│   ├── dataset1.csv              # Source dataset 1
│   ├── ... (other source files)
│   ├── dataset1_train_std.csv    # Standardized training data
│   ├── ... (other train files)
│   └── datatest.csv              # Common, standardized test set
├── model/
│   ├── *.pkl                     # Saved vectorizers and models
│   └── evaluation_results_on_common_test.json
├── static/
├── templates/
├── evaluation_reports/         # Generated .csv performance reports
├── app.py                      # Flask web application backend
├── training.py                 # Model training and evaluation script
├── evaluation_reporter.py      # Performance report generation script
├── prepare_common_test_set.py  # Data standardization script
├── requirements.txt
└── README.md
```

---

## 🤖 Model Details

* **Algorithms:** Random Forest, Multinomial Naive Bayes, and XGBoost (`XGBClassifier`).
* **Feature Extraction:** SQL queries are converted into numerical vectors using `CountVectorizer` from scikit-learn.
* **Evaluation:** All models are benchmarked against the common test set (`datatest.csv`) to ensure a fair comparison of performance.

---

## 📚 Core Libraries Used

* Flask
* Scikit-learn
* Pandas
* Joblib
* XGBoost

Specific library versions are listed in `requirements.txt`.
