import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
import tqdm
import faiss
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

class ClaimsDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        claim = self.data.iloc[idx]['summary_sentence']
        passage = self.data.iloc[idx]['text_chunk']
        group_id = self.data.iloc[idx]['book_num']
        
        claim_inputs = self.tokenizer(claim, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        passage_inputs = self.tokenizer(passage, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        return claim_inputs, passage_inputs, group_id

class GroupedSampler(Sampler):
    def __init__(self, data_source, group_ids, batch_size):
        self.data_source = data_source
        self.group_ids = group_ids
        self.batch_size = batch_size
        self.grouped_indices = self.group_indices_by_id()

    def group_indices_by_id(self):
        grouped_indices = {}
        for idx, group_id in enumerate(self.group_ids):
            if group_id not in grouped_indices:
                grouped_indices[group_id] = []
            grouped_indices[group_id].append(idx)
        return grouped_indices

    def __iter__(self):
        batches = []
        for group in self.grouped_indices.values():
            np.random.shuffle(group)
            for i in range(0, len(group), self.batch_size):
                batch = group[i:i+self.batch_size]
                if len(batch) == self.batch_size:
                    batches.append(batch)
        np.random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return len(self.data_source) // self.batch_size

from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import tqdm

def fine_tune(model, train_loader, val_loader, epochs=3, lr=5e-5, batch_size=32, device='cuda', temperature=1.0, label_smoothing=0.1, warmup_steps=2000):
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    model.to(device)

    # Initialize logging
    log_interval = 20
    training_loss_log = []
    validation_loss_log = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        step = 0

        for claim_inputs, passage_inputs, _ in tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()
            claim_inputs = {k: v.squeeze(1).to(device) for k, v in claim_inputs.items()}
            passage_inputs = {k: v.squeeze(1).to(device) for k, v in passage_inputs.items()}
            claim_outputs = model(**claim_inputs).last_hidden_state.mean(dim=1)
            passage_outputs = model(**passage_inputs).last_hidden_state.mean(dim=1)
            scores = torch.einsum("id, jd->ij", claim_outputs / temperature, passage_outputs)
            bsz = len(claim_inputs['input_ids'])
            labels = torch.arange(0, bsz, dtype=torch.long, device=device)
            loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=label_smoothing)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            
            # Log training loss every 20 steps
            if step % log_interval == 0 and step > 0:
                training_loss_log.append((epoch, step, loss.item()))
                print(f"Epoch {epoch+1}, Step {step}, Training Loss: {loss.item()}")
            step += 1

        print(f"Epoch {epoch+1}, Average Training Loss: {total_loss/len(train_loader)}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for claim_inputs, passage_inputs, _ in tqdm.tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                claim_inputs = {k: v.squeeze(1).to(device) for k, v in claim_inputs.items()}
                passage_inputs = {k: v.squeeze(1).to(device) for k, v in passage_inputs.items()}
                claim_outputs = model(**claim_inputs).last_hidden_state.mean(dim=1)
                passage_outputs = model(**passage_inputs).last_hidden_state.mean(dim=1)
                scores = torch.einsum("id, jd->ij", claim_outputs / temperature, passage_outputs)
                bsz = len(claim_inputs['input_ids'])
                labels = torch.arange(0, bsz, dtype=torch.long, device=device)
                loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=label_smoothing)
                val_loss += loss.item()
                validation_loss_log.append((epoch, step, loss.item()))
        
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss/len(val_loader)}")
    
    model.save_pretrained(f'fine_tuned_model_batch_size={batch_size}')

    # Save loss logs to files
    training_loss_filename = f'training_loss_log_batch_size_{batch_size}.txt'
    validation_loss_filename = f'validation_loss_log_batch_size_{batch_size}.txt'

    with open(training_loss_filename, 'w') as f:
        for epoch, step, loss in training_loss_log:
            f.write(f"Epoch: {epoch+1}, Step: {step}, Training Loss: {loss}\n")

    with open(validation_loss_filename, 'w') as f:
        for epoch, step, loss in validation_loss_log:
            f.write(f"Epoch: {epoch+1}, Step: {step}, Validation Loss: {loss}\n")

def main(batch_size):
    data = pd.read_csv('Data/mapped_summaries_l3.csv')
    tokenizer = AutoTokenizer.from_pretrained('fine_tuned_model_batch_size=32')
    model = AutoModel.from_pretrained('fine_tuned_model_batch_size=32')

    # Calculate the indices for the splits
    train_size = int(0.8 * len(data))
    val_size = int(0.1 * len(data))

    # Split the data linearly
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:train_size + val_size]

    train_dataset = ClaimsDataset(data=train_data, tokenizer=tokenizer)
    val_dataset = ClaimsDataset(data=val_data, tokenizer=tokenizer)

    train_group_ids = train_data['book_num'].tolist()
    train_sampler = GroupedSampler(data_source=train_dataset, group_ids=train_group_ids, batch_size=batch_size)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    
    val_group_ids = val_data['book_num'].tolist()
    val_sampler = GroupedSampler(data_source=val_dataset, group_ids=val_group_ids, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler)

    fine_tune(model, train_loader, val_loader, epochs=10, lr=5e-5, batch_size=batch_size, device=device)
    tokenizer.save_pretrained(f'fine_tuned_model_batch_size={batch_size}')

if __name__ == "__main__":
    import sys
    batch_size = int(sys.argv[1])  # Get retriever type from command line argument
    main(batch_size)
