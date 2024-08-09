import os
import random
import heapq
import tqdm
import torch
import faiss
import pandas as pd
from rank_bm25 import BM25Okapi
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer, AutoTokenizer, AutoModel
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = 'mapped_summaries_l3'

# Load your dataset
df = pd.read_csv(f"Data/{dataset}.csv")

# Define retriever classes and methods
def bm25_retriever(df):
    filtered_df = df.drop_duplicates(subset=["text_chunk"])
    tokenized_corpus = [doc.split(" ") for doc in filtered_df["text_chunk"]]
    bm25 = BM25Okapi(tokenized_corpus)

    print("Starting recall@k calculation for k=1 to 10...")

    n_correct_at_k = {k: 0 for k in range(1, 11)}
    all_doc_scores = []

    for row in tqdm.tqdm(df["summary_sentence"], desc='Precomputing scores', unit='summary'):
        tokenized_query = row.split(" ")
        doc_scores = bm25.get_scores(tokenized_query)
        all_doc_scores.append(doc_scores)

    for index, doc_scores in tqdm.tqdm(enumerate(all_doc_scores), total=len(df), desc='Recall@k', unit='row'):
        top_10_indexes = heapq.nlargest(10, range(len(doc_scores)), key=lambda i: doc_scores[i])
        retrieved_docs = [df["text_chunk"].iloc[i] for i in top_10_indexes]
        for k in range(1, 11):
            if df["text_chunk"].iloc[index] in retrieved_docs[:k]:
                n_correct_at_k[k] += 1

    recall_at_k = {k: n_correct_at_k[k] / len(df) for k in range(1, 11)}
    for k in range(1, 11):
        print(f"Recall at k = {k}: {recall_at_k[k]:.4f}")

    random_index = random.randint(0, len(df) - 1)
    random_query = df["summary_sentence"].iloc[random_index]
    random_true_doc = df["text_chunk"].iloc[random_index]
    random_top_10 = [df["text_chunk"].iloc[i] for i in heapq.nlargest(10, range(len(all_doc_scores[random_index])), key=lambda i: all_doc_scores[random_index][i])]

    print(f"Random Query: {random_query}")
    print(f"Target Document: {random_true_doc}")
    print(f"Top 10 Retrieved Documents: {random_top_10}")

def dpr_retriever(df):
    class DPRRetriever:
        def __init__(self, book_texts):
            self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
            self.context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)
            self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
            self.question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)

            self.book_texts = book_texts
            self.context_embeddings = self.encode_contexts(book_texts)

        def encode_contexts(self, texts):
            context_embeddings = []
            for text in tqdm.tqdm(texts, desc="Encoding Contexts"):
                context_input = self.context_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
                context_embedding = self.context_encoder(**context_input).pooler_output
                context_embeddings.append(context_embedding.cpu().detach().numpy())
            context_embeddings = torch.tensor(context_embeddings).squeeze(1)
            return context_embeddings

        def retrieve_passages(self, claims, top_k=10):
            batched_passages = []
            for claim in tqdm.tqdm(claims, desc="Retrieving Passages"):
                claim_input = self.question_tokenizer(claim, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
                claim_embedding = self.question_encoder(**claim_input).pooler_output

                index = faiss.IndexFlatIP(claim_embedding.size(1))
                index.add(self.context_embeddings.cpu().detach().numpy())
                _, indices = index.search(claim_embedding.cpu().detach().numpy(), top_k)

                passages = [self.book_texts[idx] for idx in indices[0]]
                batched_passages.append(passages)
            return batched_passages

    retriever = DPRRetriever(df["text_chunk"].tolist())

    recall_at_k = {k: 0 for k in range(1, 11)}
    n_correct_at_k = {k: 0 for k in range(1, 11)}
    for index, row in tqdm.tqdm(enumerate(df["summary_sentence"]), total=len(df), desc='Recall@k'):
        results = retriever.retrieve_passages([row], top_k=10)[0]
        for k in range(1, 11):
            if df["text_chunk"].iloc[index] in results[:k]:
                n_correct_at_k[k] += 1

    recall_at_k = {k: n_correct_at_k[k] / len(df) for k in recall_at_k}
    for k in range(1, 11):
        print(f"Recall at k = {k}: {recall_at_k[k]:.4f}")

    random_index = random.randint(0, len(df) - 1)
    random_query = df["summary_sentence"].iloc[random_index]
    random_true_doc = df["text_chunk"].iloc[random_index]
    random_top_10 = retriever.retrieve_passages([random_query], top_k=10)[0]

    print(f"Random Query: {random_query}")
    print(f"Target Document: {random_true_doc}")
    print(f"Top 10 Retrieved Documents: {random_top_10}")

def contriever_retriever(df, all_text, index_mapping, device='cuda'):
    # Load the fine-tuned model and tokenizer
    model = AutoModel.from_pretrained('fine_tuned_model_batch_size=32').to(device)
    tokenizer = AutoTokenizer.from_pretrained('fine_tuned_model_batch_size=32')

    def encode_texts(texts):
        embeddings = []
        for text in tqdm.tqdm(texts, desc="Encoding Texts"):
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().detach().numpy())
        return torch.tensor(embeddings).squeeze(1)

    context_embeddings = encode_texts(all_text["text_chunk"].tolist()).to(device)
    query_embeddings = encode_texts(df["summary_sentence"].tolist()).to(device)

    index = faiss.IndexFlatIP(context_embeddings.size(1))
    faiss.normalize_L2(context_embeddings.cpu().numpy())
    index.add(context_embeddings.cpu().numpy())

    recall_at_k = {k: 0 for k in range(1, 11)}

    faiss.normalize_L2(query_embeddings.cpu().numpy())
    _, indices = index.search(query_embeddings.cpu().numpy(), 10)

    n_same_book = 0
    for query_idx, retrieved_indices in enumerate(indices):
        book_index = index_mapping.get(query_idx)
        if book_index is not None:
            target_book_num = all_text.iloc[book_index]["book_num"]
            matches = sum(all_text.iloc[i]["book_num"] == target_book_num for i in retrieved_indices[:10])
            n_same_book += matches
            for k in range(1, 11):
                if book_index in retrieved_indices[:k]:
                    recall_at_k[k] += 1

    average_top_10_in_same_book = n_same_book / len(df)
    print(f"On average, {average_top_10_in_same_book} of the top 10 text chunks come from the same book")
    
    recall_at_k = {k: recall_at_k[k] / len(df) for k in recall_at_k}
    for k in range(1, 11):
        print(f"Recall at k = {k}: {recall_at_k[k]:.4f}")

def main(retriever_type):
    print(f"Doing retriever evaluation with {retriever_type} for depth 3 claims")
    df = pd.read_csv(f"Data/{dataset}.csv")
    print(f"Size of dataset: {len(df)}, running depth 3 retrieval")
    all_text = pd.read_csv("Data/all_text.csv")
    index_mapping = pd.read_csv('depth_3_index_mapping.csv').set_index('df_index')['all_text_index'].to_dict()

    if retriever_type == 'bm25':
        bm25_retriever(df)
    elif retriever_type == 'dpr':
        dpr_retriever(df)
    elif retriever_type == 'contriever':
        contriever_retriever(df, all_text, index_mapping)

if __name__ == "__main__":
    import sys
    retriever_type = sys.argv[1]  # Get retriever type from command line argument
    main(retriever_type)