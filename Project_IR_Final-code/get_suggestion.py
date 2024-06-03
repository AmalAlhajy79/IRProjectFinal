
import numpy as np
import pandas as pd
import tkinter as tk
import difflib
from searching import search


@staticmethod
def get_suggestions(self):
    if not hasattr(self, 'input_path') or not self.input_path:
        self.results_text.insert(tk.INSERT, "Please select a dataset first.\n")
        return

    query = self.query_entry.get()
    data = pd.read_json(self.input_path, lines=True)
    suggestions_with_sources = []

    if self.dataset_var.get() == "Dataset 1":
        if 'query' not in data.columns:
            self.results_text.insert(tk.INSERT, "No 'query' column found in Dataset 1.\n")
            return
        suggestions = difflib.get_close_matches(query, data['query'].tolist(), n=3)
        suggestions_with_sources = [(s, "query") for s in suggestions]

    elif self.dataset_var.get() == "Dataset 2":
        if not all(col in data.columns for col in ['text', 'metadata']):
            self.results_text.insert(tk.INSERT, "No 'text' or 'metadata' column found in Dataset 2.\n")
            return
        
        # استخراج النصوص من الأعمدة المطلوبة
        texts = data['text'].tolist()
        descriptions = [item.get('description', '') for item in data['metadata'] if isinstance(item, dict)]
        narratives = [item.get('narrative', '') for item in data['metadata'] if isinstance(item, dict)]
        
        # دمج جميع النصوص في قائمة واحدة للبحث عن الاقتراحات
        combined_texts = texts + descriptions + narratives
        
        # الحصول على الاقتراحات من جميع النصوص
        suggestions = difflib.get_close_matches(query, combined_texts, n=3)
        
        # ربط الاقتراحات مع مصادرها
        for suggestion in suggestions:
            if suggestion in texts:
                suggestions_with_sources.append((suggestion, "text"))
                # print("................here text...........")
                # print(suggestions_with_sources)
            elif suggestion in descriptions:
                suggestions_with_sources.append((suggestion, "description"))
                # print("................here description...........")
                # print(suggestions_with_sources)
            elif suggestion in narratives:
                suggestions_with_sources.append((suggestion, "narrative"))
                # print("................here narrative...........")
                # print(suggestions_with_sources)
    else:
        self.results_text.insert(tk.INSERT, "Invalid dataset selection.\n")
        return

    # عرض النتائج
    self.results_text.delete(1.0, tk.END)
    if suggestions_with_sources:
        # تنفيذ البحث باستخدام الاقتراحات
        for suggestion, _ in suggestions_with_sources:
            self.query_entry.delete(0, tk.END)
            self.query_entry.insert(0, suggestion)
            search(self)

        # حساب المقاييس بعد الاقتراحات
        overall_precision_after = np.mean([p for p in self.precision_list if p > 0])
        overall_recall_after = np.mean([r for r in self.recall_list if r > 0])
        overall_main_avg_precision_after = np.mean([ap for ap in self.average_precision_list_for_all_query if ap > 0])
        # overall_avg_precision_after = np.mean([ap for ap in self.average_precision_list if ap > 0])
        overall_reciprocal_rank_after = np.mean([rr for rr in self.reciprocal_rank_list if rr > 0])

        self.results_text.insert(tk.INSERT, f"\n\nOverall Metrics After Suggestions:\n")
        self.results_text.insert(tk.INSERT, f"Overall Precision@10: {overall_precision_after}\n")
        self.results_text.insert(tk.INSERT, f"Overall Recall@10: {overall_recall_after}\n")
        self.results_text.insert(tk.INSERT, f"Overall Mean Average Precision@10: {overall_main_avg_precision_after}\n")
        self.results_text.insert(tk.INSERT, f"Overall Reciprocal Rank@10: {overall_reciprocal_rank_after}\n")
    else:
        self.results_text.insert(tk.INSERT, "No suggestions found.\n")

