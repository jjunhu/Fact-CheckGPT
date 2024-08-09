import torch
import tqdm
from transformers import AutoModel, AutoTokenizer
import faiss
import pandas as pd

def contriever_retriever(df, all_text, index_mapping, device='cuda', fine_tuned=True):
    # Load the fine-tuned model and tokenizer
    if fine_tuned:
        model = AutoModel.from_pretrained('fine_tuned_model_batch_size=32').to(device)
        tokenizer = AutoTokenizer.from_pretrained('fine_tuned_model_batch_size=32')
    else:
        tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
        model = AutoModel.from_pretrained("facebook/contriever").to(device)
    def encode_texts(texts):
        embeddings = []
        for text in tqdm.tqdm(texts, desc="Encoding Texts"):
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().detach().numpy())
        return torch.tensor(embeddings).squeeze(1)

    context_embeddings = encode_texts(all_text['text_chunk'].tolist()).to(device)
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

if __name__ == "__main__":
    import sys
    depth = int(sys.argv[1])  # Get retriever type from command line argument
    fine_tuned = sys.argv[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if depth == 3:
        df = pd.read_csv('test_data.csv')
        index_df = pd.read_csv('depth_3_index_mapping.csv')
        index_test = index_df.tail(len(df))
        index_test.reset_index(drop=True, inplace=True)
        index_test['df_index'] = index_test.index
        index_mapping = index_test.set_index('df_index')['all_text_index'].to_dict()
    elif depth == 2:
        df = pd.read_csv('Data/test_depth_2.csv')
        index_df = pd.read_csv('depth_2_index_mapping_test.csv')
        index_mapping = index_df.set_index('df_index')['all_text_index'].to_dict()
    else:
        raise Exception("DF does not exist")
    
    all_text = pd.read_csv('Data/all_text.csv')
    print(f"Size of dataset: {len(df)}, running depth {depth} retrieval")
    
    if fine_tuned == "true":
        contriever_retriever(df, all_text, index_mapping, device='cuda')
    else:
        contriever_retriever(df, all_text, index_mapping, device='cuda', fine_tuned=False)
