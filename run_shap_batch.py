import shap 
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from typing import List, Tuple
from Bio import SeqIO
from tqdm import tqdm
from evo import Evo
from stripedhyena.tokenizer import CharLevelTokenizer
from stripedhyena.model import StripedHyena

device = 'cuda:0'

parser = argparse.ArgumentParser(description='Generate sequences using the Evo model.')
parser.add_argument('--input_negfasta', required=True, help='neg')
parser.add_argument('--input_posfasta', required=True, help='Input FASTA file path')
parser.add_argument('--model-name', type=str, default='evo-1-131k-base', help='Evo model name')
parser.add_argument('--batch-size', type=int, default=1, help='Number of sequences to evaluate at a time')
parser.add_argument('--device', type=str, default='cuda:0', help='Device for generation')
args = parser.parse_args()

class CombinedModel:
    def __init__(self, embedding_model, tokenizer, classifier):
        self.model = embedding_model
        self.tokenizer = tokenizer
        self.classifier = classifier
    '''
    def prepare_batch(self, seqs: List[str], tokenizer: CharLevelTokenizer, prepend_bos: bool = True, device: str = 'cuda:0') -> Tuple[torch.Tensor, List[int]]:
        seq_lengths = [len(seq) for seq in seqs]
        max_seq_length = max(seq_lengths)
        input_ids = []
        for seq in seqs:
            padding = [tokenizer.pad_id] * (max_seq_length - len(seq))
            input_ids.append(
                torch.tensor(
                    ([tokenizer.eod_id] * int(prepend_bos)) + tokenizer.tokenize(seq) + padding,
                    dtype=torch.long,
                ).to(device).unsqueeze(0)
            )
        input_ids = torch.cat(input_ids, dim=0)
        return input_ids, seq_lengths
    '''
    def pad_first_dimension(self, arr, m):
        w, l = arr.shape
        if w >= m:
            return arr
        padding_length = m - w
        left_padding = padding_length // 2
        right_padding = padding_length - left_padding
        padded_arr = np.pad(arr, ((left_padding, right_padding), (0, 0)), mode='constant', constant_values=0)
        return padded_arr

    def get_embeddings(self, seqs):
        class CustomEmbedding(nn.Module):
            def unembed(self, u):
                return u
        self.model.unembed = CustomEmbedding()
        tensors = []
        '''
        for i in tqdm(range(0, len(seqs), args.batch_size)):
            batch_seqs = seqs[i:i + args.batch_size]
            input_ids, seq_lengths = self.prepare_batch(batch_seqs, tokenizer, device=device, prepend_bos=True)
            with torch.inference_mode():
                embeds, _ = model(input_ids)
            for embed in embeds:
                tensor = embed[-1, :].detach().cpu().float().numpy()
                #tensor = self.pad_first_dimension(tensor, 160)
                tensors.append(tensor)
        '''
        num_batches = len(seqs) // args.batch_size + (len(seqs) % args.batch_size != 0)

        for i in range(num_batches):
            batch = seqs[i:i + args.batch_size]
            batch = torch.from_numpy(batch).to(device)
            with torch.inference_mode():
                embeds, _ = model(batch)
            for embed in embeds:
                tensor = embed[-1, :].detach().cpu().float().numpy()
                #tensor = self.pad_first_dimension(tensor, 160)
                tensors.append(tensor)
        torch.cuda.empty_cache()
        tensors = np.stack(tensors, axis=0)
        tensors = tensors.reshape(tensors.shape[0], -1)
        return tensors

    def predict(self, text_list):
        embeddings = self.get_embeddings(text_list)
        return self.classifier.predict(embeddings)
    def predict_proba(self, text_list):
        embeddings = self.get_embeddings(text_list)
        return self.classifier.predict_proba(embeddings)

pos_seqs = [str(record.seq) for record in SeqIO.parse(args.input_posfasta, 'fasta')]
neg_seqs = [str(record.seq) for record in SeqIO.parse(args.input_negfasta, 'fasta')]
seqs = pos_seqs + neg_seqs
evo_model = Evo('evo-1-131k-base')
model, tokenizer = evo_model.model, evo_model.tokenizer
model.to(device)
model.eval()
torch.cuda.empty_cache()


def prepare_batch(seqs: List[str], tokenizer: CharLevelTokenizer, prepend_bos: bool = True, device: str = 'cuda:0') -> Tuple[torch.Tensor, List[int]]:
        seq_lengths = [len(seq) for seq in seqs]
        max_seq_length = max(seq_lengths)
        input_ids = []
        for seq in seqs:
            padding = [tokenizer.pad_id] * (max_seq_length - len(seq))
            input_ids.append(
                torch.tensor(
                    ([tokenizer.eod_id] * int(prepend_bos)) + tokenizer.tokenize(seq) + padding,
                    dtype=torch.long,
                ).to(device).unsqueeze(0)
            )
        input_ids = torch.cat(input_ids, dim=0)
        return input_ids, seq_lengths

y_positive = np.ones(len(pos_seqs))
y_negative = np.zeros(len(neg_seqs))
y = np.concatenate((y_positive, y_negative))
nan_indices = np.isnan(y)
y = y[~nan_indices]
seqs = np.array(seqs)
seqs = seqs[~nan_indices]

#convert to input_ids
input_ids, seq_lengths = prepare_batch(seqs, tokenizer, device=device, prepend_bos=True)




#seqs = seqs.tolist()

X_train, X_test, y_train, y_test = train_test_split(input_ids.cpu().numpy(), y, test_size=0.3, random_state=42)
print(type(X_train))
clf = LogisticRegression(max_iter=1000)
combined_model = CombinedModel(model, tokenizer, clf)
X_train_embeddings = combined_model.get_embeddings(X_train)
clf.fit(X_train_embeddings, y_train)
y_pred = combined_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Custom wrapper for SHAP
class ModelWrapper:
    def __init__(self, combined_model):
        self.combined_model = combined_model
    
    def predict(self, X):
        return self.combined_model.predict_proba(X)[:, 1]  # Assuming binary classification

# Use the wrapper for SHAP
wrapped_model = ModelWrapper(combined_model)

# Perform SHAP analysis
background = X_train[:10]  # Use a smaller subset for the background dataset
print(background.shape)
explainer = shap.KernelExplainer(wrapped_model.predict, background)

# Batch SHAP calculation
batch_size = 5
num_batches = len(X_test) // batch_size + (1 if len(X_test) % batch_size != 0 else 0)
shap_values = []

for i in range(num_batches):
    batch_X_test = X_test[i*batch_size:(i+1)*batch_size]
    batch_shap_values = explainer.shap_values(batch_X_test)
    shap_values.append(batch_shap_values)

shap_values = np.concatenate(shap_values, axis=1)

# SHAP summary plot
shap.summary_plot(shap_values, X_test)
plt.savefig('shap_summary_plot.png')
