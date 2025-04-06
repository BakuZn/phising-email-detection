Phishing Email Detection:

This project implements a machine learning model to detect phishing or spam emails using Natural Language Processing (NLP) techniques and a Naive Bayes classifier.

Dataset
The dataset spam.csv includes:

sender, receiver, date, subject, body, label, and urls

label:

1 → Spam

0 → Not Spam

Features
Text preprocessing: lowercasing, punctuation removal, stopword removal, stemming

Feature extraction using CountVectorizer

Classification using Multinomial Naive Bayes

Model evaluation with accuracy score and classification report

Model Performance
Achieved 94% accuracy on the test dataset

How to Run
Ensure Python is installed.

Install required packages:

nginx
Copy
Edit
pip install pandas nltk scikit-learn
Place spam.csv and main.py in the same directory.

Run the program.
Output
Displays sample data

Shows vectorized shape

Prints model accuracy and classification report
