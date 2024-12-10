"""
Sample from a trained model
"""
import os
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT
import numpy as np

import torch.nn.functional as F
from tqdm.auto import tqdm
from torcheval.metrics.text import Perplexity

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
seed = 1337
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
weight_tying = False
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf, weight_tying=weight_tying)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

import torch

def calculate_isotropy(W):
    """
    Calculate the isotropy measure I(W) for a given word embedding matrix W.
    
    Args:
        W (torch.Tensor): A tensor of shape (vocab_size, emb_dim) representing the word embeddings.

    Returns:
        float: The isotropy measure I(W).
    """
    vocab_size, emb_dim = W.shape  # Get dimensions of the embedding matrix
    assert emb_dim == 128, f'expected to have emb_dim=128, but got {emb_dim}'
    
    # Step 1: Compute the covariance matrix W^T W
    covariance_matrix = torch.mm(W.T, W)
    
    # Step 2: Perform eigenvalue decomposition of W^T W using torch.linalg.eig
    eigenvalues, eigenvectors = torch.linalg.eig(covariance_matrix)
    
    # Eigenvectors will be complex, so we need to take the real part for further calculations
    eigenvectors = eigenvectors.real
    
    # Assert that eigenvectors have expected dimensions (emb_dim, emb_dim)
    assert eigenvectors.shape == (emb_dim, emb_dim), f"Expected eigenvectors to have shape ({emb_dim}, {emb_dim}), but got {eigenvectors.shape}"
    
    # Assert that W has shape (vocab_size, emb_dim)
    assert W.shape == (vocab_size, emb_dim), f"Expected W to have shape ({vocab_size}, {emb_dim}), but got {W.shape}"
    
    # Step 3: Compute the partition function Z(a) for all eigenvectors at once
    # We do this by calculating exp(W @ eigenvectors) in one go.
    exp_values = torch.exp(torch.matmul(W, eigenvectors))
    
    # Assert that exp_values has expected dimensions (vocab_size, emb_dim)
    assert exp_values.shape == (vocab_size, emb_dim), f"Expected exp_values to have shape ({vocab_size}, {emb_dim}), but got {exp_values.shape}"
    
    # Step 4: Sum along the token (vocab) dimension to get Z(a) for each eigenvector a
    Z_values = exp_values.sum(dim=0)  # Z(a) for each eigenvector
    
    # Assert that Z_values has length emb_dim, corresponding to the number of eigenvectors
    assert Z_values.shape[0] == emb_dim, f"Expected Z_values to have length {emb_dim}, but got {Z_values.shape[0]}"
    
    # Step 5: Compute I(W) as the ratio of the minimum to maximum Z(a)
    Z_min = Z_values.min().item()
    Z_max = Z_values.max().item()
    
    # Step 6: Compute isotropy
    isotropy = Z_min / Z_max if Z_max != 0 else 0
    
    return isotropy

@torch.no_grad()
def estimate_metrics(logits_and_targets):
    
    out = {}
    k = 5  # For Recall@k
    languages = ['big', 'small']
    metrics = ['accuracy', 'recall@5', 'mrr']

    out = {}
    for language in languages:
        out[language] = {}
        for metric_name in metrics:
            out[language][metric_name] = 0
        

    # Loop over both ('val', 'big') and ('val', 'small')
    for language in tqdm(languages, desc='lang'):
        accuracies = []
        recalls_at_k = []
        mrrs = []

        # Get all validation data using get_all_val_data generator
        num_batches = len(logits_and_targets[language]['logits'])
        for i in tqdm(range(num_batches), desc='batch'):
            logits = logits_and_targets[language]['logits'][i]
            Y = logits_and_targets[language]['targets'][i]
            # Calculate accuracy
            predictions = logits.argmax(dim=-1)  # Shape: (batch_size, seq_len)
            valid_mask = (Y != -1)  # Mask to ignore -1 indices
            correct = (predictions == Y) & valid_mask
            accuracy = correct.sum().float() / valid_mask.sum().float()
            accuracies.append(accuracy.item())

            # Calculate rank of each target in the logits
            sorted_logits = logits.argsort(dim=-1, descending=True)  # Shape: (batch_size, seq_len, vocab_size)
            target_ranks = torch.where(sorted_logits == Y.unsqueeze(-1))[2] + 1  # 1-based rank

            # Apply valid mask to ranks
            target_ranks = target_ranks.view(-1)[valid_mask.view(-1)]

            # Calculate Recall@k
            recall_k = (target_ranks <= k).float().mean().item()
            recalls_at_k.append(recall_k)

            # Calculate MRR
            reciprocal_ranks = 1.0 / target_ranks.float()
            mrr = reciprocal_ranks.mean().item()
            mrrs.append(mrr)
        # Stack results and compute the mean for this split and language
        out[language]['accuracy'] = torch.tensor(accuracies).mean().item()
        out[language]['recall@5'] = torch.tensor(recalls_at_k).mean().item()
        out[language]['mrr'] = torch.tensor(mrrs).mean().item()
    for lang in languages:
        for metric_name in metrics:
            print(f'language: {lang}, metrics: {metric_name}: {out[lang][metric_name]}')
    return out

print('I(W) (unembeddings):')
print('High-resource language embeddings:', calculate_isotropy(model.lm_head.weight[:65]))
print('Low-resource language embeddings:', calculate_isotropy(model.lm_head.weight[65:]))
print('all embeddings:', calculate_isotropy(model.lm_head.weight))

print('I(W) (embeddings):')
print('High-resource language embeddings:', calculate_isotropy(model.transformer.wte.weight[:65]))
print('Low-resource language embeddings:', calculate_isotropy(model.transformer.wte.weight[65:]))
print('all embeddings:', calculate_isotropy(model.transformer.wte.weight))
print('---')
# Load the dictionary from the specified path
results_path = os.path.join(out_dir, 'eval_results.pth')
logits_and_targets = torch.load(results_path)
print("Results loaded successfully.")
print('---')

estimate_metrics(logits_and_targets)
print('---')

def calculate_and_save_ppls(logits_and_targets):
    Ts = np.array([i/100 for i in range(1, 201)])
    Ts2PPL = {}
    for lang in ['small', 'big']:
        Ts2PPL[lang] = []
        logits_list = logits_and_targets[lang]['logits']
        Y_list = logits_and_targets[lang]['targets']
        for T in tqdm(Ts):
            metric=Perplexity(ignore_index=-1)
            for logits, Y in zip(logits_list, Y_list):
                new_logits = logits * 1/T
                metric.update(new_logits, Y)
            loss_T = metric.compute()
            Ts2PPL[lang].append(loss_T.item())
        Ts2PPL[lang] = np.array(Ts2PPL[lang])
        am = Ts2PPL[lang].argmin()
        print(f'lang: {lang}, PPL(T=1): {Ts2PPL[lang][99]}, PPL_min(T={Ts[am]}): {Ts2PPL[lang][am]}')
    return Ts2PPL

ppls = calculate_and_save_ppls(logits_and_targets)
print('---')
ppls_path = os.path.join(out_dir, 'ppls.pth')
torch.save(ppls, ppls_path)
print(f"PPLs saved to {ppls_path}")