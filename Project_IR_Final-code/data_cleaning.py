import re
import os
import json
import tkinter as tk
from tkinter import ttk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

class DataCleaning:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        result = []

        for word in words:
            if word.lower() not in self.stop_words and re.search(r'\b(under|of|without|in|on|for|at|by|is|are|to|with|from|the|a)\b', word) is None:
                lemmed_word = self.lemmatizer.lemmatize(word.lower())
                result.append(lemmed_word)

        return ' '.join(result)

def load_and_cleaning_dataset(input_path):
    with open(input_path, 'r') as file:
        data = file.readlines()

    cleaning_class = DataCleaning()
    cleaned_data = []
    raw_data = []

    for one_text in data:
        cleaned_text = cleaning_class.clean_text(one_text)
        if cleaned_text:
            cleaned_data.append(cleaned_text)
            raw_data.append(one_text)

    return cleaned_data, raw_data

def cleaning_and_representation_dataset(app):
    # if not hasattr(app, 'input_path') or not app.input_path:
    #     app.results_text.insert(tk.INSERT, "Please select a dataset first.\n")
    #     return

    cleaned_data, raw_data = load_and_cleaning_dataset(app.input_path)

    if not cleaned_data:
        app.results_text.insert(tk.INSERT, "Loaded dataset is empty after cleaning.\n")
        return

    corpus_data = cleaned_data

    final_results = []

    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus_data)

    for doc_id, doc_val in enumerate(corpus_data):
        vector = vectorizer.transform([doc_val]).toarray()
        distribution = vector[0]
        if not all(v == 0 for v in distribution):
            final_results.append({
                "id_doc": doc_id,
                "vector_doc": vector.tolist(),
                "distribution_doc": distribution.tolist()
            })

    path_of_outer_dir = 'C:/Users/AMAL/Desktop/Project_IR_Final'  # تعديل المسار هنا

    if not os.path.exists(path_of_outer_dir):
        os.makedirs(path_of_outer_dir)

    output_file_path = os.path.join(path_of_outer_dir, 'represent_dataset.json')

    with open(output_file_path, 'w') as output_file:
        json.dump(final_results, output_file, indent=4)
    print("Data cleaned and represented successfully.\n")
    print(f"Saved to {output_file_path}\n")

    app.vectorizer = vectorizer
    app.tfidf_matrix = vectorizer.transform(corpus_data)
    app.ids_data = [json.loads(line).get('_id', f"doc{i}") for i, line in enumerate(raw_data)]