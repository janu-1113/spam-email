# Spam Email Classification

A machine-learning project that classifies emails as **spam** or **ham (not spam)** using the **Enron-Spam dataset** and the **Naive Bayes classifier** from NLTK.

## Project Overview

This project processes email text and trains a Naive Bayes model to distinguish spam emails from legitimate emails.

The notebook follows these main steps:

1. Load spam and ham email files from the dataset.
2. Preprocess email text using tokenization, lowercasing, stop-word removal, and lemmatization.
3. Label spam emails as `1` and ham emails as `0`.
4. Convert each email into a word-frequency feature dictionary.
5. Split the data into training and testing sets.
6. Train and evaluate an NLTK Naive Bayes classifier.

## Dataset

The project uses the **Enron-Spam dataset**.

Dataset information and download:
http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/index.html

The notebook expects the dataset to be arranged like this:

```text
project-folder/
├── spam_email_classification.ipynb
├── README.md
└── data/
    └── enron1/
        ├── spam/
        │   ├── email-file-1
        │   └── ...
        └── ham/
            ├── email-file-1
            └── ...
```

The `spam` folder contains spam emails, while the `ham` folder contains legitimate emails. The notebook reads files from `./data/enron1/spam` and `./data/enron1/ham`.

## Technologies and Libraries

- Python
- NLTK — text processing and Naive Bayes classification
- scikit-learn — imported in the notebook (the `shuffle` utility is not used in the final workflow)
- Python standard libraries: `os`, `collections`, and `random`

## Installation

1. Clone or download this project.
2. Install the required Python packages:

   ```bash
   pip install nltk scikit-learn notebook
   ```

3. Download and extract the Enron-Spam dataset.
4. Place the `enron1` dataset folder inside the project's `data` folder, following the structure shown above.
5. Open the notebook:

   ```bash
   jupyter notebook spam_email_classification.ipynb
   ```

## NLTK Resources

The notebook uses NLTK's tokenizer and WordNet lemmatizer. If the required resources are not already installed, run the following in a Python cell:

```python
import nltk

nltk.download("punkt")
nltk.download("wordnet")

# Some newer NLTK versions may also require this tokenizer resource:
nltk.download("punkt_tab")
```

## How It Works

### 1. Load the emails

Email files are read using ISO-8859-1 encoding. Spam and ham emails are loaded into separate lists.

### 2. Preprocess the text

The `preprocess_sentence()` function:

- Tokenizes each email using `nltk.word_tokenize()`.
- Converts tokens to lowercase.
- Removes the stop words defined in the notebook.
- Filters a small set of the least frequent words in each email.
- Applies WordNet lemmatization.

### 3. Label the data

Each spam email is assigned label `1`; each ham email is assigned label `0`. The combined samples are shuffled before feature extraction.

### 4. Extract features

Each email is represented as a dictionary in which each word is a feature and its value is the word's frequency.

### 5. Split the dataset

The notebook uses a **70% training / 30% testing** split.

### 6. Train and evaluate

An NLTK Naive Bayes classifier is trained using the training set. The notebook reports accuracy on both the training set and the test set.

## Run the Project

Run the cells in `spam_email_classification.ipynb` from top to bottom after installing the dependencies and placing the dataset in the expected directory.

The final notebook cell prints:

- Training accuracy
- Testing accuracy

The exact accuracy values depend on the dataset files and the shuffled train/test split.

## Project Files

| File / Folder | Description |
|---|---|
| `spam_email_classification.ipynb` | Notebook containing data loading, preprocessing, feature extraction, training, and evaluation |
| `data/enron1/spam/` | Spam email text files |
| `data/enron1/ham/` | Ham email text files |
| `README.md` | Project documentation |

## Notes

- The notebook uses relative paths, so run it from the project folder or update the dataset paths.
- The split is performed after shuffling, but no fixed random seed is set. Results can therefore vary between runs.
- The notebook reports accuracy; it does not currently calculate precision, recall, F1-score, or a confusion matrix.
