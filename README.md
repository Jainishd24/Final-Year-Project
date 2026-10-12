# ⚖️ Judgment Law AI
### AI-Powered Bail Prediction and Judicial Case Analysis



Judgment Law AI is an academic machine learning project that explores the use of artificial intelligence and natural language processing to analyze judicial case information and predict bail outcomes. The project combines a trained machine learning model, structured case data, processed case analysis, and an interactive Streamlit interface.

The system is designed to demonstrate how machine learning can be applied to legal-text analysis and how model predictions can be presented through an accessible web application.

> **Disclaimer:** This project is developed for academic and research purposes only. Predictions are experimental and must not be treated as legal advice, an actual judicial decision, or a substitute for qualified legal counsel.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Objectives](#-objectives)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation and Setup](#-installation-and-setup)
- [Running the Application](#-running-the-application)
- [How It Works](#-how-it-works)
- [Dataset and Model](#-dataset-and-model)
- [Limitations](#-limitations)
- [Future Enhancements](#-future-enhancements)
- [License](#-license)
- [Connect with me](#-connect-with-me)

## 📖 Project Overview

Understanding judicial decisions requires examining case facts, legal arguments, relevant circumstances, and judicial reasoning. This project explores whether machine learning techniques can identify patterns in historical case data and use them to generate an experimental bail-outcome prediction.

The application provides an interactive interface for presenting case information and displaying the model's predicted outcome. It also uses structured case data and processed JSON files to support case analysis and exploration.

The project is intended to support learning, experimentation, and discussion about the applications and limitations of AI in the legal domain.

## 🎯 Objectives

- Develop an interactive application for exploring legal case information.
- Apply a trained machine learning model to bail-outcome prediction.
- Use historical case data as input for experimentation and analysis.
- Present predictions and relevant case information through a user-friendly interface.
- Explore the potential applications of machine learning in legal technology.
- Understand the limitations, reliability concerns, and ethical implications of AI-assisted legal analysis.

## ✨ Key Features

- **Interactive web application:** A browser-based interface developed with Streamlit.
- **Machine learning inference:** Uses a previously trained and saved model to generate predictions.
- **Bail outcome classification:** Displays a predicted outcome, such as *Granted* or *Rejected*, according to the model's output mapping.
- **Structured dataset integration:** Loads case records from a CSV file.
- **Processed case analysis:** Includes JSON files containing processed case information.
- **Case exploration:** Supports presenting available case details and prediction results through the interface.
- **Reusable model:** Loads a saved model rather than retraining it every time the application starts.
- **Academic research focus:** Provides a foundation for studying machine learning applications in judicial case analysis.

*Features shown here describe the current project structure and intended application. Only features implemented and verified in the code should be considered operational.*

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Web application | Streamlit |
| Data manipulation | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Model serialization | Joblib |
| Dataset format | CSV |
| Case analysis format | JSON |
| Data processing utilities | Python |

## 🏗️ System Architecture

The application's high-level workflow is:

1. **Data input:** Load case records and relevant information.
2. **Data preparation:** Use the available case fields required by the saved model.
3. **Model loading:** Load the serialized machine learning pipeline.
4. **Prediction:** Pass supported input features to the model.
5. **Result presentation:** Display the predicted bail outcome in the Streamlit interface.
6. **Case exploration:** Present available case details and processed analysis where supported.

```text
Historical Case Data
        |
        v
CSV Dataset and Case Records
        |
        v
Data Preparation
        |
        v
Saved ML Model
(bail_prediction_model.joblib)
        |
        v
Bail Outcome Prediction
        |
        v
Streamlit Web Interface
        |
        v
Prediction and Case Information
```

## 📁 Project Structure

```text
judgment-law-ai/
│
├── app.py
│   └── Main Streamlit application
│
├── all_cases.csv
│   └── Case dataset used by the project
│
├── bail_prediction_model.joblib
│   └── Saved machine learning pipeline
│
├── final_results/
│   └── Processed case analysis in JSON format
│
├── utils/
│   ├── excel_export.py
│   └── regex_clean.py
│
├── requirements.txt
│   └── Python dependency list
│
├── .gitignore
│   └── Excludes local environments and unnecessary files
│
├── LICENSE
│   └── Project license
│
└── README.md
    └── Project documentation
```

## 💻 Prerequisites

Before running the project, ensure you have:

- Python 3.10 or another Python version compatible with the listed dependencies and saved model.
- pip, the Python package installer.
- Git, to clone the repository.
- A modern web browser.
- Sufficient disk space for the dataset, model, and Python dependencies.

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Jainishd24/Final-Year-Project.git
```

### 2. Navigate to the project directory

```bash
cd Final-Year-Project
```

### 3. Create a virtual environment

**Windows — PowerShell:**

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the appropriate execution-policy settings for your environment or activate the environment from Command Prompt instead.

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If installation fails, verify that your Python version is compatible with the packages and the serialized model.

## ▶️ Running the Application

From the project root, run:

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal, typically:

```text
http://localhost:8501
```

Open that URL in your browser to access the application.

To stop the application, press `Ctrl + C` in the terminal.

## 🔍 How It Works

The application uses a saved machine learning pipeline to make predictions from the input features it expects.

At a high level:

1. The application loads the dataset and saved model.
2. The user explores or selects the case information supported by the interface.
3. The application prepares the required model inputs.
4. The saved pipeline generates a prediction.
5. The interface presents the result for educational analysis.

The prediction reflects patterns learned from the model's training data. It does not establish whether bail should legally be granted or rejected in any particular case.

## 📊 Dataset and Model

### Dataset

The repository includes `all_cases.csv`, a structured dataset used by the application. Processed case-analysis files are stored in `final_results/`.

Before reusing or redistributing any case data, verify its source, applicable usage terms, and whether the information has been handled appropriately.

### Saved model

The file `bail_prediction_model.joblib` contains the serialized machine learning pipeline used for inference.

The current implementation uses the following input fields:

- `text`
- `court`
- `gender`

The application's output mapping interprets the model's class `1` as **Granted** and class `0` as **Rejected**. This mapping describes the implementation; it does not establish that the prediction is accurate.

### Evaluation

A saved model and a functioning interface do not, by themselves, demonstrate predictive reliability. Any reported accuracy, precision, recall, F1-score, or other performance metric should be supported by a documented evaluation using appropriate held-out data.

## ⚠️ Limitations

- **Experimental predictions:** Outputs are model-generated estimates, not judicial decisions.
- **Data quality:** Historical case records may contain incomplete, inconsistent, or biased information.
- **Potential bias:** A model can reproduce patterns or biases present in its training data.
- **Generalization:** Performance on new cases, courts, or jurisdictions may differ from performance on the original dataset.
- **Explainability:** A predicted label does not necessarily explain the legal reasoning behind an outcome.
- **Legal context:** Bail decisions depend on case-specific facts, applicable law, judicial discretion, and other relevant circumstances.
- **Evaluation requirements:** Model performance must be independently measured before claims about accuracy or practical reliability can be made.

**Do not use this application to make real legal, judicial, bail, or liberty-related decisions.** It is not a substitute for a qualified legal professional.

## 🔮 Future Enhancements

Potential directions for further development include:

- Comprehensive model evaluation and error analysis.
- Improved data validation and preprocessing.
- Explainability tools to help examine the factors associated with model predictions.
- Better handling of missing and inconsistent case information.
- More robust testing across different courts and case types.
- Clearer documentation of dataset provenance and model-training methodology.
- Improved accessibility, user experience, and deployment options.
- Ethical and bias assessments before any proposed real-world application.

These are potential enhancements, not claims that the features are already implemented.

### Academic Information

- **Project title:** Judgment Law AI
- **Project type:** Collaborative final-year academic project
- **Institution:** SVKM'S NMIMS Mukesh Patel School Of Technology, Management and Engineering 
- **Department:** Computer Science 

## 📜 License

This project is intended to use the **MIT License**, subject to agreement from the team members who hold rights in the contributed work.

The MIT License permits use, modification, and redistribution subject to its conditions, including preservation of the copyright and permission notice. See the [`LICENSE`](LICENSE) file for the complete terms.

Third-party libraries, datasets, and other external materials may have separate license terms. The project license does not override those terms.

## Connect with me 
- **Linkedin:** https://www.linkedin.com/in/jainish-devpura/
- **Portfolio:** https://portfolio-theta-ruddy-65.vercel.app/
- **Email ID:** jainishdevpura24@gmail.com
---

