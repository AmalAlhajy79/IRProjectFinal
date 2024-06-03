import os
import json
import tkinter as tk
from tkinter import scrolledtext, ttk
from tkinter import scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from data_cleaning import cleaning_and_representation_dataset
from indexing_dataset import indexing_datataset
from searching import search
from get_suggestion import get_suggestions
from search_Use_advanced_word_embedding_models import search_Use_advanced_word_embedding_models


class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")


#relevant_docs = {}
class Application(tk.Tk):
    input_path = None  # تعديل: إضافة متغير input_path كمتغير ثابت

    def __init__(self):
        super().__init__()

        self.title("..... Information Retrieval System")
        self.geometry("900x700")
        self.configure(bg='#FFFFFF')  # لون أبيض
        self.relevant_docs = {}
        self.create_widgets()

    def create_widgets(self):
        # إعدادات الألوان
        bg_color = '#FFFFFF'  # لون أبيض
        btn_color = '#d17a93'  # لون زهري نصف غامق
        btn_hover_color = '#b36479'
        entry_bg_color = '#ffe4e1'  # لون زهر نصف فاتح
        text_bg_color = '#ffe4e1'  # لون زهر نصف فاتح
        label_color = '#000000'  # أسود
        results_fg_color = '#000000'  # أسود

        # إطار للتمرير
        container = tk.Frame(self, bg=bg_color)
        container.pack(fill="both", expand=True)

        top_frame = tk.Frame(container, bg=bg_color)
        top_frame.pack(fill="x", pady=10)

        middle_frame = tk.Frame(container, bg=bg_color)
        middle_frame.pack(fill="x", pady=10)

        bottom_frame = tk.Frame(container, bg=bg_color)
        bottom_frame.pack(fill="x", pady=10)

        # قائمة اختيار مجموعة البيانات
        self.dataset_var = tk.StringVar(self)
        self.dataset_var.set("Select Dataset")
        self.dataset_menu = tk.OptionMenu(top_frame, self.dataset_var, "Dataset 1", "Dataset 2", command=self.load_dataset)
        self.dataset_menu.configure(bg=btn_color, fg='white', activebackground=btn_hover_color, font=('Helvetica', 12))
        self.dataset_menu.pack(side="right", padx=5)

        # إدخال الاستعلام
        self.query_label = tk.Label(middle_frame, text="Enter your query:", font=('Helvetica', 12), bg=bg_color, fg=label_color)
        self.query_label.pack(side="left", padx=5)

        self.query_entry = tk.Entry(middle_frame, bg=entry_bg_color, fg=label_color, font=('Helvetica', 12))
        self.query_entry.pack(side="left", padx=5)

        self.search_button = tk.Button(middle_frame, text="Search", command=lambda:search(self), bg=btn_color, fg='white', activebackground=btn_hover_color, font=('Helvetica', 12))
        self.search_button.pack(side="left", padx=5)

        # زر جلب الاقتراحات
        self.suggestions_button = tk.Button(middle_frame, text="query suggestions", command=lambda:get_suggestions(app), bg=btn_color, fg='white', activebackground=btn_hover_color, font=('Helvetica', 12))
        self.suggestions_button.pack(side="left", padx=5)


        #زر للكلمات المتشابهة 
        self.suggestions_button = tk.Button(middle_frame, text="Use advanced word embedding models ", command=lambda:search_Use_advanced_word_embedding_models(app), bg=btn_color, fg='white', activebackground=btn_hover_color, font=('Helvetica', 12))
        self.suggestions_button.pack(side="right", padx=5)

        # نتائج البحث
        self.results_text = scrolledtext.ScrolledText(bottom_frame, width=70, height=15, bg=text_bg_color, fg=results_fg_color, font=('Helvetica', 12))
        self.results_text.pack(pady=10, padx=10, fill="x")

        # زر حساب القياسات الإجمالية
        self.calculate_metrics_button = tk.Button(bottom_frame, text="Calculate all Metrics", command=self.calculate_overall_metrics, bg=btn_color, fg='white', activebackground=btn_hover_color, font=('Helvetica', 12))
        self.calculate_metrics_button.pack(pady=5)

        # المساحة المخصصة للقياسات
        self.metrics_text = tk.Text(bottom_frame, height=10, width=70, bg=text_bg_color, fg=results_fg_color, font=('Helvetica', 12))
        self.metrics_text.pack(pady=10, padx=10, fill="x")

        self.vectorizer = None
        self.tfidf_matrix = None
        self.ids_data = None
        from data_cleaning import DataCleaning
        self.cleaning_class =DataCleaning()

    def load_dataset(self, dataset_choice):
        if dataset_choice == "Dataset 1":
            self.input_path = 'E:/lotte/lifestyle/dev/qas.search.jsonl'
            cleaning_and_representation_dataset(self)
            indexing_datataset(self)

        elif dataset_choice == "Dataset 2":
            self.input_path = 'E:/webis-touche2020/queries.jsonl'
            cleaning_and_representation_dataset(self)
            indexing_datataset(self)
        else:
            return

        self.results_text.insert(tk.INSERT, f"Selected {dataset_choice} Enter your query now ...\n")

        self.queries_searched = []
        self.precision_list = []
        self.recall_list = []
        self.average_precision_list = []
        self.reciprocal_rank_list = []
        self.average_precision_list_for_all_query=[]
        self.relevant_docs.clear()  #....update add self
        #update
        self.precision_list_word2vec = []
        self.recall_list_word2vec = []
        self.average_precision_list_word2vec = []
        self.reciprocal_rank_list_word2vec = []
        #----------------------------------------------------
    def calculate_overall_metrics(self):
        precision_values = [p for p in self.precision_list if p > 0]
        recall_values = [r for r in self.recall_list if r > 0]
        avg_precision_values = [ap for ap in self.average_precision_list_for_all_query if ap > 0]
        reciprocal_rank_values = [rr for rr in self.reciprocal_rank_list if rr > 0]

        if not precision_values or not recall_values or not avg_precision_values or not reciprocal_rank_values:
            self.metrics_text.insert(tk.INSERT, "No search queries executed yet.\n")
            return

        overall_precision = np.mean(precision_values)
        overall_recall = np.mean(recall_values)
        overall_avg_precision = np.mean(avg_precision_values)
        overall_reciprocal_rank = np.mean(reciprocal_rank_values)

        self.metrics_text.insert(tk.INSERT, "Overall Metrics:\n")
        self.metrics_text.insert(tk.INSERT, f"Overall Precision@10: {overall_precision}\n")
        self.metrics_text.insert(tk.INSERT, f"Overall Recall@10: {overall_recall}\n")
        self.metrics_text.insert(tk.INSERT, f"Overall Main Average Precision@10: {overall_avg_precision}\n")
        self.metrics_text.insert(tk.INSERT, f"Overall Reciprocal Rank@10: {overall_reciprocal_rank}\n")

if __name__ == "__main__":
    app = Application()
    app.mainloop()
