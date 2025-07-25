# Fake News Detector

A simple machine learning app that classifies news as real or fake using Scikit-learn and Streamlit.

## Features

- Enter or upload a news article
- Predicts if the news is real or fake
- Built with TfidfVectorizer and PassiveAggressiveClassifier
- Clean Streamlit user interface

## Dataset

This project uses the [Fake or Real News Dataset](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news) from Kaggle, containing over 44,000 labeled news articles.

## Future Plan

This project will be upgraded to analyze and detect the authenticity of **current news headlines** from live sources.

## Installation

```bash
git clone https://github.com/Longman-max/fake-news-ml.git
cd fake-news-ml
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run app/app.py
```

## Project Structure

```
fake-news-ml/
│
├── app/               # Streamlit UI
├── data/              # Dataset
├── models/            # Trained model
├── train_model.py     # Training script
├── requirements.txt
└── README.md
```

## How It Works

- The input text is transformed using `TfidfVectorizer`
- The classifier used is `PassiveAggressiveClassifier`
- Trained on labeled articles to detect patterns and make predictions

## License

MIT License
