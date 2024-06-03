

import numpy as np
import pandas as pd
import tkinter as tk
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_cleaning import DataCleaning,load_and_cleaning_dataset


def search_Use_advanced_word_embedding_models(self):
    if self.vectorizer is None or self.tfidf_matrix is None:
        self.results_text.insert(tk.INSERT, "Please load a dataset first.\n")
        return

    query = self.query_entry.get()
    self.queries_searched.append(query)

    cleaned_query = self.cleaning_class.clean_text(query)
    query_vector = self.vectorizer.transform([cleaned_query])

    scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
    sorted_indices = np.argsort(scores)[::-1]

    filtered_results = [(self.ids_data[idx], scores[idx]) for idx in sorted_indices if scores[idx] > 0.0]
    top_10_results = filtered_results[:10]
    results = [doc_id for doc_id, score in top_10_results]

    if query not in self.relevant_docs and results:
        self.relevant_docs[query] = results

    relevant_documents = self.relevant_docs.get(query, [])

    # Load the dataset to retrieve document contents
    data = pd.read_json(self.input_path, lines=True)

    # Determine the appropriate columns for ID and content
    id_column = None
    content_column = None
    metadata_column = 'metadata'  # Assuming 'metadata' column exists
    answer_pids_column = 'answer_pids'

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
    
    # حساب المقاييس قبل استخدام Word2Vec
    precision_at_10 = sum(1 for doc_id in results if doc_id in relevant_documents) / 10
    self.precision_list.append(precision_at_10)
     
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
   
    # recall_at_10 = sum(1 for doc_id in results if doc_id in relevant_documents) / len(filtered_results) if relevant_documents else 0
    # self.recall_list.append(recall_at_10)

    # relevant_count = 0
    # average_precision_at_10 = 0
    # for i, doc_id in enumerate(results, 1):
    #     if doc_id in relevant_documents:
    #         relevant_count += 1
    #         average_precision_at_10 += relevant_count / i
    # average_precision_at_10 /= min(len(relevant_documents), 10) if relevant_documents else 1
    # self.average_precision_list_for_all_query.append(average_precision_at_10)
    # # self.average_precision_list.append(average_precision_at_10)

    reciprocal_rank = 0
    for i, doc_id in enumerate(results, 1):
        if doc_id in relevant_documents:
            reciprocal_rank = 1 / i
            break
    self.reciprocal_rank_list.append(reciprocal_rank)

    overall_precision_before = np.mean(self.precision_list) if self.precision_list else 0
    overall_recall_before = np.mean(self.recall_list) if self.recall_list else 0
    overall_avg_precision_before = np.mean(self.average_precision_list_for_all_query) if self.average_precision_list_for_all_query else 0
    overall_reciprocal_rank_before = np.mean(self.reciprocal_rank_list) if self.reciprocal_rank_list else 0

    self.metrics_text.delete(1.0, tk.END)
    self.metrics_text.insert(tk.INSERT, f"\nOverall Metrics Before Using Word2Vec:\n")
    self.metrics_text.insert(tk.INSERT, f"Overall Precision@10: {overall_precision_before}\n")
    self.metrics_text.insert(tk.INSERT, f"Overall Recall@10: {overall_recall_before}\n")
    self.metrics_text.insert(tk.INSERT, f"Overall Mean Average Precision@10: {overall_avg_precision_before}\n")
    self.metrics_text.insert(tk.INSERT, f"Overall Reciprocal Rank@10: {overall_reciprocal_rank_before}\n\n")

    # بناء نموذج Word2Vec
    cleaned_data, raw_data = load_and_cleaning_dataset(self.input_path)
    documents = cleaned_data
    word2vec_model = Word2Vec([doc.split() for doc in documents], vector_size=300, window=10, min_count=2, sg=1)

    # تحسين عملية تحويل النصوص إلى متجهات
    def get_word2vec_vector(text, model):
        words = text.split()
        word_vecs = [model.wv[word] for word in words if word in model.wv]
        if len(word_vecs) == 0:  # تجنب القيم الفارغة
            return np.zeros(model.vector_size)
        return np.mean(word_vecs, axis=0)

    query_vec_word2vec = get_word2vec_vector(cleaned_query, word2vec_model)
    similarities_word2vec = []
    for doc in documents:
        doc_vec_word2vec = get_word2vec_vector(doc, word2vec_model)
        similarity = cosine_similarity([query_vec_word2vec], [doc_vec_word2vec])[0][0]
        similarities_word2vec.append(similarity)

    sorted_indices_word2vec = np.argsort(similarities_word2vec)[::-1]
    filtered_results_word2vec = [(self.ids_data[idx], similarities_word2vec[idx]) for idx in sorted_indices_word2vec if similarities_word2vec[idx] > 0.0]
    top_10_results_word2vec = filtered_results_word2vec[:10]

    self.results_text.insert(tk.INSERT, "\nTop 10 Relevant Documents Using Word2Vec:\n\n")
    for doc_id, score in top_10_results_word2vec:
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

    # حساب المقاييس بعد استخدام Word2Vec
    precision_at_10_word2vec = sum(1 for doc_id, score in top_10_results_word2vec if doc_id in relevant_documents) / 10
    self.precision_list_word2vec.append(precision_at_10_word2vec)

    recall_at_10_word2vec = sum(1 for doc_id, score in top_10_results_word2vec if doc_id in filtered_results) / len(relevant_documents) if relevant_documents else 0
    self.recall_list_word2vec.append(recall_at_10_word2vec)




    relevant_count_word2vec = 0
    average_precision_at_10 = 0
    sum_score=0
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa")
    for i, score in top_10_results:
            relevant_count_word2vec += 1
            sum_score+=score
            average_precision_at_10=sum_score/relevant_count_word2vec
    self.average_precision_list_for_all_query.append(average_precision_at_10)

    print("......relevant_count_word2vec......")
    print(relevant_count_word2vec)
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
   

    relevant_count_word2vec = 0
    sum_score_word2vec=0
    average_precision_at_10_word2vec = 0
    for i, (doc_id, score) in enumerate(top_10_results_word2vec, 1):
            if doc_id in relevant_documents:
               relevant_count_word2vec += 1
               sum_score_word2vec +=score
               average_precision_at_10_word2vec=sum_score_word2vec/relevant_count_word2vec
        # if doc_id in relevant_documents:
        #     relevant_count_word2vec += 1
        #     average_precision_at_10_word2vec += relevant_count_word2vec / i
    # average_precision_at_10_word2vec /= min(len(relevant_documents), 10) if relevant_documents else 1
    self.average_precision_list_word2vec.append(average_precision_at_10_word2vec)




    reciprocal_rank_word2vec = 0
    for i, (doc_id, score) in enumerate(top_10_results_word2vec, 1):
        if doc_id in relevant_documents:
            reciprocal_rank_word2vec = 1 / i
            break
    self.reciprocal_rank_list_word2vec.append(reciprocal_rank_word2vec)

    overall_precision_after = np.mean(self.precision_list_word2vec) if self.precision_list_word2vec else 0
    overall_recall_after = np.mean(self.recall_list_word2vec) if self.recall_list_word2vec else 0
    overall_avg_precision_after = np.mean(self.average_precision_list_word2vec) if self.average_precision_list_word2vec else 0
    overall_reciprocal_rank_after = np.mean(self.reciprocal_rank_list_word2vec) if self.reciprocal_rank_list_word2vec else 0

    self.metrics_text.insert(tk.INSERT, f"\nOverall Metrics After Using Word2Vec:\n")
    self.metrics_text.insert(tk.INSERT, f"Overall Precision@10: {overall_precision_after}\n")
    self.metrics_text.insert(tk.INSERT, f"Overall Recall@10: {overall_recall_after}\n")
    self.metrics_text.insert(tk.INSERT, f"Overall Mean Average Precision@10: {overall_avg_precision_after}\n")
    self.metrics_text.insert(tk.INSERT, f"Overall Reciprocal Rank@10: {overall_reciprocal_rank_after}\n\n")

    print(f"Precision before: {overall_precision_before}, after: {overall_precision_after}")
    print(f"Recall before: {overall_recall_before}, after: {overall_recall_after}")
    print(f"Mean Average Precision before: {overall_avg_precision_before}, after: {overall_avg_precision_after}")
    print(f"Reciprocal Rank before: {overall_reciprocal_rank_before}, after: {overall_reciprocal_rank_after}")


# from gensim.models import Word2Vec
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import sys
# sys.stdout.reconfigure(encoding='utf-8')

# # قائمة الوثائق
# documents = [
#     "apple banana pic fruit",
#     "orange fruit",
#     "apple pic  orange pens fruit pensel",
#     "mango pi fruit",
#     "banana pens mango"
# ]

# # تحويل النصوص إلى تصورات TF-IDF
# tfidf_vectorizer = TfidfVectorizer()
# tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# # حساب التشابه بين الاستعلام والوثائق
# query = "pens"
# query_vec = tfidf_vectorizer.transform([query])
# cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

# # فرز الوثائق حسب التشابه
# related_docs_indices = cosine_similarities.argsort()[::-1]
# print("نتائج البحث باستخدام TF-IDF:")
# for index in related_docs_indices:
#     print(documents[index])

# # بناء نموذج Word2Vec
# word2vec_model = Word2Vec([doc.split() for doc in documents], vector_size=100, window=5, min_count=1, sg=1)

# # تحويل النصوص إلى تصورات Word2Vec
# word_vectors = word2vec_model.wv

# # حساب التشابه بين الاستعلام والوثائق باستخدام Word2Vec
# query_vec = word_vectors[query]
# similarities = []
# for doc in documents:
#     doc_vec = sum([word_vectors[word] for word in doc.split()])
#     similarity = cosine_similarity([query_vec], [doc_vec])[0][0]
#     similarities.append(similarity)

# # فرز الوثائق حسب التشابه
# related_docs_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
# print("نتائج البحث بعد استخدام Word2Vec:")
# for index in related_docs_indices:
#     print(documents[index])
