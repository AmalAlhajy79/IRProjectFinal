import re
import os
import json
import tkinter as tk

def indexing_datataset(self):

        index_dir = 'C:/Users/AMAL/Desktop/Project_IR_Final/indexing_Folder'
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)

        index_file_path = os.path.join(index_dir, 'indexing.json')

        index_data = {
            "ids_data": self.ids_data,
            "tfidf_matrix": self.tfidf_matrix.toarray().tolist()
        }

        with open(index_file_path, 'w') as index_file:
            json.dump(index_data, index_file, indent=4)

        print("Dataset indexed successfully..\n")
        print(f"Saved to {index_file_path}\n")

        # self.queries_searched = []
        # self.precision_list = []
        # self.recall_list = []
        # self.average_precision_list = []
        # self.reciprocal_rank_list = []

        # self.relevant_docs.clear()#------add self
