# Phishing Email Detection System

A command-line tool that uses machine learning (scikit-learn) and text
analysis to classify emails as **phishing** or **legitimate**, storing
every analysis result in a local SQLite database.

Built for: B207 Cyber Security — Idea 1 (Network Intrusion / Phishing
Email Detection with Machine Learning).

## How it works

1. **Preprocessing** (`preprocess.py`) — extracts subject/body text
   from `.eml` files, plain `.txt` files, or raw CLI input, and cleans
   the text (HTML stripping, lower-casing, whitespace normalisation).
2. **Feature extraction** (`features.py`) —
   - TF-IDF vectorization of the cleaned text (unigrams + bigrams).
   - Hand-crafted phishing indicator features: number of links,
     suspicious top-level domains, IP-address links, urgency language
     ("act now", "24 hours", "suspended"), action words ("click here",
     "verify your account"), generic greetings, requests for sensitive
     information, exclamation marks, uppercase ratio, text length.
3. **Model** (`model.py`) — combines both feature sets and trains a
   **Random Forest** classifier (Logistic Regression also supported)
   using scikit-learn. The trained pipeline is saved with `joblib`.
4. **Database** (`database.py`) — SQLite storage for every analyzed
   email, its prediction/confidence, extracted features, detected
   phishing attempts, and model training metadata.
5. **CLI** (`main.py`) — command-line interface tying it all together.

## 1. Environment setup (one script does everything)

```bash
chmod +x setup.sh
./setup.sh
```

This creates a virtual environment, installs all dependencies from
`requirements.txt`, and trains the model on the bundled sample dataset
(`data/sample_emails.csv`), which also creates and seeds the SQLite
database (`phishing_detection.db`).

If you prefer to do it manually:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train_model.py
```

On Windows PowerShell, use:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
```

## 2. Using the tool

Activate the environment first: `source venv/bin/activate`

### Train / retrain the model

```bash
python main.py train
python main.py train --algorithm logistic_regression
```

### Analyze an email

From a `.eml` or `.txt` file:
```bash
python main.py analyze --file sample_emails/phishing_sample.eml
python main.py analyze --file sample_emails/legitimate_sample.eml
```

From subject + body typed directly:
```bash
python main.py analyze --subject "Verify your account now" \
                        --body "Click here http://fake-bank.tk immediately or your account will be suspended."
```

From raw text (no separate subject):
```bash
python main.py analyze --text "Urgent! Your package could not be delivered, click here to reschedule."
```

Each analysis prints the prediction, a confidence score, and the list
of extracted indicator features, and is saved to the database.

### View results

```bash
python main.py list             # recent analysis results
python main.py phishing         # detected phishing attempts only
python main.py stats            # overall statistics + latest model metrics
python main.py report                       # print a text report
python main.py report --output report.txt   # save the report to a file
```

## Project structure

```
phishing_detector/
├── setup.sh                 # one script: venv + deps + train + init DB
├── requirements.txt
├── main.py                  # CLI entry point
├── train_model.py           # training script
├── model.py                 # ML pipeline (TF-IDF + handcrafted features + classifier)
├── features.py               # handcrafted phishing-indicator feature extraction
├── preprocess.py             # email loading & text cleaning
├── database.py                # SQLite schema + queries
├── data/
│   └── sample_emails.csv     # labeled training dataset (phishing / legitimate)
├── sample_emails/
│   ├── phishing_sample.eml
│   └── legitimate_sample.eml
└── phishing_detection.db     # created on first run
```

## Notes on the dataset

`data/sample_emails.csv` contains 41 hand-written labeled examples
(20 phishing, 21 legitimate) for demonstration purposes. For a stronger
model, replace/extend this file with a larger public phishing-email
dataset (e.g. from Kaggle) — the CSV format (`label,subject,body`) is
all `train_model.py` requires.

## Evaluation metrics

`train_model.py` reports accuracy, precision, recall, F1 score, and a
confusion matrix on a held-out test split, and stores them in the
`model_metadata` table so `python main.py stats` can always show the
latest model's performance.
