"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import numpy as np
import torch
import tiktoken
from tqdm.auto import tqdm
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
seed = 1337
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
weight_tying=True
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

def get_all_val_data(dataset, block_size, batch_size, language='small', vocab_size=65):
    """
    Fetch all validation data for a selected language, skipping incomplete final batch.
    
    Args:
        dataset (str): The name of the dataset (e.g., 'shakespeare_char').
        block_size (int): The size of each input sequence block.
        batch_size (int): Number of sequences per batch.
        language (str): Specify 'big' or 'small' for the language.
        vocab_size (int): The size of the vocabulary.
    
    Returns:
        A generator that yields batches of (x, y) tensors.
    """
    
    # Define paths and load validation data
    data_dir = os.path.join('data', dataset)
    data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    # Prepare to return data in chunks of batch_size
    for i in range(0, len(data) - block_size * batch_size, block_size * batch_size):
        x_list = []
        y_list = []

        for b in range(batch_size):
            start_idx = i + b * block_size
            end_idx = start_idx + block_size

            if end_idx > len(data):
                break  # Skip incomplete data at the end

            # Prepare input (x) and target (y) sequences
            x_sample = torch.from_numpy((data[start_idx:end_idx]).astype(np.int64))
            y_sample = torch.from_numpy((data[start_idx + 1:end_idx + 1]).astype(np.int64))

            # Modify x and y if the language is 'small'
            if language == 'small':
                x_sample = x_sample + vocab_size
                y_sample = y_sample + vocab_size

            x_list.append(x_sample)
            y_list.append(y_sample)

        # Stack the sequences into tensors
        x = torch.stack(x_list)
        y = torch.stack(y_list)

        x, y = x.to(device), y.to(device)

        # Yield the current batch
        yield x, y

@torch.no_grad()
def get_logits_and_targets(model, block_size, batch_size, dataset='shakespeare_char'):
    """
    Calculate logits for (val, small) and (val, big)
    using get_all_val_data, stacking results from multiple batches and averaging them.
    
    Args:
        block_size (int): Block size of input sequences.
        batch_size (int): Batch size (number of sequences per batch).
    
    Returns:
        out (dict): Dictionary containing logits and targets.
    """

    languages = ['big', 'small']
    out = {}
    for language in languages:
        out[language] = {}
        out[language]['logits'] = []
        out[language]['targets'] = []
    for language in tqdm(languages, desc='lang'):

        # Get all validation data using get_all_val_data generator
        for i, (X, Y) in tqdm(
            enumerate(get_all_val_data(dataset, block_size, batch_size, language=language)),
            desc='batch'
        ):
            with torch.no_grad():
                logits, _ = model.forward_eval(X, Y)  # logits shape: (batch_size, seq_len, vocab_size)
            out[language]['logits'].append(logits)
            out[language]['targets'].append(Y)
        
    return out


batch_size = 12
block_size = model.config.block_size

out = get_logits_and_targets(model, block_size, batch_size)
results_path = os.path.join(out_dir, 'eval_results.pth')
# Save the dictionary to the specified path
torch.save(out, results_path)
print(f"Results saved to {results_path}")