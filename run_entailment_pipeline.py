import os
import json
import torch
import tqdm
import pandas as pd
import heapq
import faiss
from transformers import AutoTokenizer, pipeline, DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer, AutoModel
from rank_bm25 import BM25Okapi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = 'evaluation_entailment'

class BM25Retriever:
    def __init__(self, book_texts):
        tokenized_corpus = [doc.split(" ") for doc in book_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.book_texts = book_texts

    def retrieve_passages(self, claims, top_k=10):
        batched_passages = []
        for claim in tqdm.tqdm(claims, desc="Retrieving Passages"):
            tokenized_query = claim.split(" ")
            doc_scores = self.bm25.get_scores(tokenized_query)
            top_k_indexes = heapq.nlargest(top_k, range(len(doc_scores)), key=lambda i: doc_scores[i])
            passages = [self.book_texts[i] for i in top_k_indexes]
            batched_passages.append(passages)
        return batched_passages

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

# class ColbertRetriever:
#     def __init__(self, book_texts):
#         doc_maxlen = 1024
#         nbits = 16  
#         checkpoint = 'downloads/colbertv2.0'
#         index_name = f'{dataset}.{nbits}bits'
#         collection = book_texts

#         with Run().context(RunConfig(nranks=4, experiment='notebook')):
#             config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits)

#             indexer = Indexer(checkpoint=checkpoint, config=config)
#             indexer.index(name=index_name, collection=collection, overwrite=True)

#         self.indexer = indexer.get_index()  # Get the absolute path of the index, if needed.

#         with Run().context(RunConfig(experiment='notebook')):
#             self.searcher = Searcher(index=index_name)

#     def retrieve_passages(self, claims, top_k=5):
#         batched_passages = []
#         rankings = self.searcher.search_all(claims, k=top_k).todict()
#         for index, results in rankings.items():
#             passages = [self.searcher.collection[pid] for pid, _, _ in results]
#             batched_passages.append(passages)
#         return batched_passages

class ContrieverRetriever:
    def __init__(self, book_texts):
        self.model = AutoModel.from_pretrained('fine_tuned_model_batch_size=32').to(device)
        self.tokenizer = AutoTokenizer.from_pretrained('fine_tuned_model_batch_size=32')
        self.context_embeddings = self.encode_texts(book_texts)
        self.book_texts = book_texts

    def encode_texts(self, texts):
        embeddings = []
        for text in tqdm.tqdm(texts, desc="Encoding Texts"):
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().detach().numpy())
        return torch.tensor(embeddings).squeeze(1)

    def retrieve_passages(self, claims, top_k=10):
        query_embeddings = self.encode_texts(claims)
        index = faiss.IndexFlatIP(self.context_embeddings.size(1))
        index.add(self.context_embeddings.numpy())
        _, indices = index.search(query_embeddings.numpy(), top_k)

        batched_passages = []
        for retrieved_indices in indices:
            passages = [self.book_texts[idx] for idx in retrieved_indices]
            batched_passages.append(passages)
        return batched_passages

class EntailmentChecker:
    def __init__(self, type):
        model_name = "facebook/bart-large-mnli" if type == "bart" else "FacebookAI/roberta-large-mnli"
        self.classifier = pipeline("text-classification", model=model_name, device=device)
        self.tokenizer = self.classifier.tokenizer
        self.max_length = self.tokenizer.model_max_length

    def check_entailment(self, claims, batched_passages, gold_chunks, entailments):
        results = []
        
        for claim, passages, gold_chunk, entailment in tqdm.tqdm(zip(claims, batched_passages, gold_chunks, entailments), total=len(claims), desc="Processing Claims"):
            claim_results = []
            for passage in passages[:10]:  # considering top k=10 passages
                claim_length = len(self.tokenizer.encode(claim, add_special_tokens=True))
                max_premise_length = self.max_length - claim_length - 1
                premise_encoded = self.tokenizer.encode(passage, add_special_tokens=True, truncation=True, max_length=max_premise_length)
                truncated_premise = self.tokenizer.decode(premise_encoded, skip_special_tokens=True)
                
                result = self.classifier(f"{truncated_premise}{self.tokenizer.sep_token}{claim}") 
                predicted_label = result[0]['label'].lower()
                print(predicted_label)
                
                claim_results.append({
                    "passage": passage,
                    "entailment": predicted_label
                })
            
            # Checking entailment for the gold nugget
            gold_claim_length = len(self.tokenizer.encode(claim, add_special_tokens=True))
            max_gold_premise_length = self.max_length - gold_claim_length - 1
            gold_premise_encoded = self.tokenizer.encode(gold_chunk, add_special_tokens=True, truncation=True, max_length=max_gold_premise_length)
            truncated_gold_premise = self.tokenizer.decode(gold_premise_encoded, skip_special_tokens=True)
            
            gold_result = self.classifier(f"{truncated_gold_premise} {self.tokenizer.sep_token} {claim}")
            gold_predicted_label = gold_result[0]['label'].lower()
            
            results.append({
                "claim": claim,
                "retrieved_passages": claim_results,
                "target_passage": gold_chunk,
                "gold_entailment": gold_predicted_label,
                "target_entailment": entailment
            })
                        
        return results


class EntailmentPipeline:
    def __init__(self, book_texts, entailment_type, retriever_type):
        retriever_class = {
            'bm25': BM25Retriever,
            'dpr': DPRRetriever,
            'contriever': ContrieverRetriever
        }[retriever_type]

        self.retriever = retriever_class(book_texts)
        self.entailment_checker = EntailmentChecker(entailment_type)

    def process_claims(self, claims, entailments, gold_chunks):
        batched_passages = self.retriever.retrieve_passages(claims, top_k=10)
        entailment_results = self.entailment_checker.check_entailment(claims, batched_passages, gold_chunks, entailments)
        return entailment_results

def main(entailment_type, retriever_type):
    print(f"Running entailment with entailment model: {entailment_type}, retriever model: {retriever_type}")
    # Load dataset
    df = pd.read_csv(f"Data/{dataset}.csv")
    all_text = pd.read_csv(f"Data/all_text.csv")

    # Unique book texts
    unique_book_texts = all_text["text_chunk"].tolist()
    claims = df["summary_sentence"].tolist()
    entailments = df["Entailment"].tolist()
    gold_chunks = df["text_chunk"].tolist()

    # Initialize the pipeline
    pipeline = EntailmentPipeline(unique_book_texts, entailment_type, retriever_type)

    # Process claims
    results = pipeline.process_claims(claims, entailments, gold_chunks)

    # Save results to a JSON file
    output_file = f"results_{retriever_type}_{entailment_type}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    import sys
    entailment_type = sys.argv[1]  # Get entailment type from command line argument
    retriever_type = sys.argv[2]  # Get retriever type from command line argument
    main(entailment_type, retriever_type)
