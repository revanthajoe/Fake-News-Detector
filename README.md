```markdown
# Fake News Detector

## Overview

This repository implements a simple fake-news detection pipeline (see fk.ipynb). The notebook:

- Loads two CSV files (fake.csv and True.csv) containing news articles.
- Labels fake articles with 0 and real articles with 1, concatenates and shuffles the dataset.
- Removes unused columns (title, subject, date) and keeps the article text and label.
- Cleans text using the wordopt(text) preprocessing function (lowercasing, remove URLs, HTML tags, punctuation, digits, and newlines).
- Vectorizes text using TfidfVectorizer and trains a LogisticRegression classifier.
- Uses a 70/30 train/test split and evaluates the model on held-out data.

## Dataset

Place the dataset files in a `data/` directory at the repository root with the following filenames:

- data/fake.csv  (fake news examples)
- data/True.csv  (real news examples)

You can also download the dataset from the Google Drive folder:
https://drive.google.com/drive/folders/1Gt7P7DoMnDQn09v1zVqK7yueisS8MWse?usp=drive_link

Notes:
- The notebook originally referenced absolute paths (e.g., `R:\Personal\College\Ai - COE\Project\fake.csv`). Before running, update those paths to the relative `data/` paths above or adjust to your environment.

## Installation / Dependencies

Tested with Python 3.8+. Install dependencies:

pip install -r requirements.txt

If requirements.txt is not present, at minimum install:

pip install pandas numpy scikit-learn jupyter

## Usage

Open and run the notebook `fk.ipynb` from top to bottom. High-level steps executed by the notebook:

1. Load `data/fake.csv` and `data/True.csv`.
2. Assign labels (`fake` -> 0, `True` -> 1) and concatenate the dataframes.
3. Shuffle and reset indices.
4. Preprocess text using the `wordopt(text)` function (lowercase, strip URLs/HTML/punctuation/digits, remove newlines).
5. Split into train and test sets with `train_test_split(test_size=0.3)`.
6. Vectorize text using `TfidfVectorizer()`:
   - `.fit_transform()` on training texts
   - `.transform()` on test texts
7. Train a `LogisticRegression()` classifier on the vectorized training data.
8. Evaluate model performance on the test set (accuracy / other metrics you choose to compute).
9. (Optional) Export the trained model with joblib or pickle for later inference.

Tips:
- To speed up experiments, pass `max_features` or `min_df` to `TfidfVectorizer`.
- Consider adding stopword removal or lemmatization/stemming to `wordopt` if needed.

## Preprocessing details

The notebook defines:

- wordopt(text):  
  - Lowercases text  
  - Removes URLs and "www" links  
  - Removes HTML tags  
  - Removes punctuation and digits  
  - Replaces newline characters with spaces

You can extend preprocessing with stopword removal, stemming/lemmatization, or custom token filters.

## Model

Current configuration in the notebook:

- Feature extraction: `sklearn.feature_extraction.text.TfidfVectorizer`
- Classifier: `sklearn.linear_model.LogisticRegression`
- Train/test split: 70% train / 30% test

Feel free to try other models (Naive Bayes, SVM, tree-based models, or neural networks) and other vectorizers (CountVectorizer, word embeddings).

## Reproducing results

- Ensure the dataset files are in `data/` and open `fk.ipynb`.
- Run all cells from top to bottom.
- If you want to save the trained model, add joblib/pickle export lines at the end of the notebook.

## Contribution

Contributions are welcome. To contribute:

- Open an issue describing the change or bug.
- Fork the repository and create a branch for your work.
- Add code or notebooks demonstrating the change if applicable.
- Submit a pull request with a clear description of the change.

Suggested improvements:
- Replace absolute file paths in fk.ipynb with relative `data/` paths.
- Add a `requirements.txt`.
- Add a script or module version of training for non-notebook usage (e.g., scripts/train.py).
- Add evaluation metrics and model persistence (joblib).

## License

This project is distributed under the MIT License.
```
