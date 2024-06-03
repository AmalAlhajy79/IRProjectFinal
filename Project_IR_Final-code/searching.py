

#الكود الاساسي الشغال
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import tkinter as tk

#import sys
# sys.stdout.reconfigure(encoding='utf-8')
def search(self):
    if self.vectorizer is None or self.tfidf_matrix is None:
        self.results_text.insert(tk.INSERT, "Please load a dataset first.\n")
        return


    #استقبال الكويري.......
    query = self.query_entry.get()
    self.queries_searched.append(query)
     
    #تنظيف الكويري.......
    cleaned_query = self.cleaning_class.clean_text(query)
    query_vector = self.vectorizer.transform([cleaned_query])

    scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
    sorted_indices = np.argsort(scores)[::-1]

    filtered_results = [(self.ids_data[idx], scores[idx]) for idx in sorted_indices if scores[idx] > 0.0]
    print("........filtered_results....document that score > 0 ...")
    print(filtered_results)
    top_10_results = filtered_results[:10]
    results = [doc_id for doc_id, score in top_10_results]

    if query not in self.relevant_docs and results:
        self.relevant_docs[query] = results

    relevant_documents = self.relevant_docs.get(query, [])

    precision_at_10 = sum(1 for doc_id in results if doc_id in relevant_documents) / 10
    self.precision_list.append(precision_at_10)

    recall_at_10 = sum(1 for doc_id in results if doc_id in relevant_documents) / len(filtered_results) if relevant_documents else 0
    self.recall_list.append(recall_at_10)


    relevant_count = 0
    average_precision_at_10 = 0
    sum_score=0
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa")
    for i, score in top_10_results:
            relevant_count += 1
            sum_score+=score
            average_precision_at_10=sum_score/relevant_count
    self.average_precision_list_for_all_query.append(average_precision_at_10)

    print("......relevant_count......")
    print(relevant_count)
    print("......sum_score......")
    print(sum_score)
    print("......average_precision_at_10......")
    print(average_precision_at_10)
    print("......average_precision_list_for_all_query......")
    print(self.average_precision_list_for_all_query)

    count_map=0
    sum_values=0
    for idx, value in enumerate(self.average_precision_list_for_all_query, start=1):
        count_map+=1
        sum_values+=value
    map_10=sum_values/count_map
    print(f"....map@_10....")  
    print(map_10)
   

    reciprocal_rank = 0
    for i, doc_id in enumerate(results, 1):
        if doc_id in relevant_documents:
            reciprocal_rank = 1 / i
            break
    self.reciprocal_rank_list.append(reciprocal_rank)

    # Load the dataset to retrieve document contents
    data = pd.read_json(self.input_path, lines=True)

    # Determine the appropriate columns for ID and content
    id_column = None
    content_column = None
    metadata_column = 'metadata'  # Assuming 'metadata' column exists
    answer_pids_column='answer_pids'

    if 'qid' in data.columns:
        id_column = 'qid'
        content_column = 'query'
    elif '_id' in data.columns and 'text' in data.columns:
        id_column = '_id'
        content_column = 'text'
    else:
        self.results_text.insert(tk.INSERT, "ID or content column not found in the dataset.\n")
        return

    self.results_text.delete(1.0, tk.END)
    self.results_text.insert(tk.INSERT, "Top 10 Relevant Documents :\n\n")

    for doc_id, score in top_10_results:
        if id_column == 'qid':
            doc_id_numeric = int(''.join(filter(str.isdigit, doc_id)))
            doc_rows = data.loc[data[id_column] == doc_id_numeric]
        else:
            doc_id = int(doc_id) if doc_id.isdigit() else doc_id  # Ensure the id is in correct format
            doc_rows = data.loc[data[id_column] == doc_id]

        if doc_rows.empty:
            self.results_text.insert(tk.INSERT, f"ID: {doc_id}\nScore: {score:.4f}\nContent: Not found\n\n")
        else:
            if 'qid' in data.columns:
                id_column = 'qid'
                content_column = 'query'
                doc_content = doc_rows[content_column].values[0]
                anser_pids = doc_rows[answer_pids_column].values[0]
                self.results_text.insert(tk.INSERT, f"ID_doc : {doc_id}\n query : {doc_content}\n anser_pids : {anser_pids}\n\n")
                print(f"Score in doc1: {score:.4f}\n")
            else:   
                doc_content = doc_rows[content_column].values[0] 
                metadata = doc_rows[metadata_column].values[0]
                description = metadata.get('description', 'No description available')
                narrative = metadata.get('narrative', 'No narrative available')
                self.results_text.insert(tk.INSERT, f"ID : {doc_id}\nText : {doc_content}\nDescription : {description}\nNarrative : {narrative}\n\n")
                print(f"Score in doc2: {score:.4f}\n")

        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.INSERT, f"\n Precision@10: {precision_at_10}\n")
        self.metrics_text.insert(tk.INSERT, f"Recall@10: {recall_at_10}\n")
        self.metrics_text.insert(tk.INSERT, f"Average Precision@10: {average_precision_at_10}\n")
        self.metrics_text.insert(tk.INSERT, f"MAP@10: {map_10}\n")
        self.metrics_text.insert(tk.INSERT, f"Reciprocal Rank@10: {reciprocal_rank}\n\n")














#الكود الاساسي الشغال
# def search(self):
#     if self.vectorizer is None or self.tfidf_matrix is None:
#         self.results_text.insert(tk.INSERT, "Please load a dataset first.\n")
#         return


#     #استقبال الكويري.......
#     query = self.query_entry.get()
#     self.queries_searched.append(query)
     
#     #تنظيف الكويري.......
#     cleaned_query = self.cleaning_class.clean_text(query)
#     query_vector = self.vectorizer.transform([cleaned_query])

#     scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
#     sorted_indices = np.argsort(scores)[::-1]

#     filtered_results = [(self.ids_data[idx], scores[idx]) for idx in sorted_indices if scores[idx] > 0.0]
#     print("........filtered_results....document that score > 0 ...")
#     print(filtered_results)
#     top_10_results = filtered_results[:10]
#     results = [doc_id for doc_id, score in top_10_results]

#     if query not in self.relevant_docs and results:
#         self.relevant_docs[query] = results

#     relevant_documents = self.relevant_docs.get(query, [])

#     precision_at_10 = sum(1 for doc_id in results if doc_id in relevant_documents) / 10
#     self.precision_list.append(precision_at_10)

#     recall_at_10 = sum(1 for doc_id in results if doc_id in relevant_documents) / len(relevant_documents) if relevant_documents else 0
#     self.recall_list.append(recall_at_10)

#     relevant_count = 0
#     average_precision_at_10 = 0
#     for i, doc_id in enumerate(results, 1):
#         if doc_id in relevant_documents:
#             relevant_count += 1
#             average_precision_at_10 += relevant_count / i
#     average_precision_at_10 /= min(len(relevant_documents), 10) if relevant_documents else 1
#     self.average_precision_list.append(average_precision_at_10)

#     reciprocal_rank = 0
#     for i, doc_id in enumerate(results, 1):
#         if doc_id in relevant_documents:
#             reciprocal_rank = 1 / i
#             break
#     self.reciprocal_rank_list.append(reciprocal_rank)

#     # Load the dataset to retrieve document contents
#     data = pd.read_json(self.input_path, lines=True)

#     # Determine the appropriate columns for ID and content
#     id_column = None
#     content_column = None
#     metadata_column = 'metadata'  # Assuming 'metadata' column exists
#     answer_pids_column='answer_pids'

#     if 'qid' in data.columns:
#         id_column = 'qid'
#         content_column = 'query'
#     elif '_id' in data.columns and 'text' in data.columns:
#         id_column = '_id'
#         content_column = 'text'
#     else:
#         self.results_text.insert(tk.INSERT, "ID or content column not found in the dataset.\n")
#         return

#     self.results_text.delete(1.0, tk.END)
#     self.results_text.insert(tk.INSERT, "Top 10 Relevant Documents :\n\n")

#     for doc_id, score in top_10_results:
#         if id_column == 'qid':
#             doc_id_numeric = int(''.join(filter(str.isdigit, doc_id)))
#             doc_rows = data.loc[data[id_column] == doc_id_numeric]
#         else:
#             doc_id = int(doc_id) if doc_id.isdigit() else doc_id  # Ensure the id is in correct format
#             doc_rows = data.loc[data[id_column] == doc_id]

#         if doc_rows.empty:
#             self.results_text.insert(tk.INSERT, f"ID: {doc_id}\nScore: {score:.4f}\nContent: Not found\n\n")
#         else:
#             if 'qid' in data.columns:
#                 id_column = 'qid'
#                 content_column = 'query'
#                 doc_content = doc_rows[content_column].values[0]
#                 anser_pids = doc_rows[answer_pids_column].values[0]
#                 self.results_text.insert(tk.INSERT, f"ID_doc : {doc_id}\n query : {doc_content}\n anser_pids : {anser_pids}\n\n")
#                 print(f"Score in doc1: {score:.4f}\n")
#             else:   
#                 doc_content = doc_rows[content_column].values[0] 
#                 metadata = doc_rows[metadata_column].values[0]
#                 description = metadata.get('description', 'No description available')
#                 narrative = metadata.get('narrative', 'No narrative available')
#                 self.results_text.insert(tk.INSERT, f"ID : {doc_id}\nText : {doc_content}\nDescription : {description}\nNarrative : {narrative}\n\n")
#                 print(f"Score in doc2: {score:.4f}\n")

#         self.metrics_text.delete(1.0, tk.END)
#         self.metrics_text.insert(tk.INSERT, f"\n Precision@10: {precision_at_10}\n")
#         self.metrics_text.insert(tk.INSERT, f"Recall@10: {recall_at_10}\n")
#         self.metrics_text.insert(tk.INSERT, f"Average Precision@10: {average_precision_at_10}\n")
#         self.metrics_text.insert(tk.INSERT, f"Reciprocal Rank@10: {reciprocal_rank}\n\n")
