import math
import torch
from torch.nn import functional as F
from model import GPT

import torch.nn as nn
from model import Block, LayerNorm

class GradLogger:
    def __init__(self, train_steps, vocab_size):
        self.type2history = dict()
        self.type2step = dict()
        for k in [
                    "emb",
                    "grad_input",
                    "grad_output_positive",
                    "grad_output_negative",
                    "grad_total",
                    "grad_adam_total",
                    "grad_adam_moments",
                    "grad_adam_decay",
                    "output_loss"
                ]:
            self.type2history[k] = torch.zeros((train_steps, vocab_size), dtype=torch.float32)
            self.type2step[k] = 0
        self.emb_prev = None

    def set_history(self, k, value):
        current_step = self.type2step[k]
        self.type2history[k][current_step] = value
        self.type2step[k] += 1

    def get_history(self, k, i):
        return self.type2history[k][i]

    def set_emb_prev(self, emb_prev):
        self.emb_prev = emb_prev.clone().detach()

    def get_emb_prev(self):
        return self.emb_prev

    def save_state_dict(self, state_dict_path):
        """
        Save type2history to the specified path as a state_dict.
        """
        # Prepare a state dict with all the histories
        state_dict = {k: v.clone() for k, v in self.type2history.items()}
        
        # Save to the specified path
        torch.save(state_dict, state_dict_path)

class EmbeddingWithLogger(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, grad_logger):
        super().__init__(num_embeddings, embedding_dim)
        self.grad_logger = grad_logger

    def forward(self, input):
        def my_hook(grad):
            self.grad_logger.set_history('grad_input', torch.norm(grad, dim=1))
            return grad

        # make a copy of weights to register hook on it
        identity_matrix = torch.eye(self.weight.size(0), device=self.weight.device)
        weight = torch.matmul(identity_matrix, self.weight).requires_grad_(True)
        weight.register_hook(my_hook)
        return weight[input]


class LinearWithLogger(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, grad_logger=None, grad_type=None):
        super().__init__(in_features, out_features, bias)
        self.grad_logger = grad_logger
        self.grad_type = grad_type

    def forward(self, input):
        def my_hook(grad):
            self.grad_logger.set_history(self.grad_type, torch.norm(grad, dim=1))
            return grad

        # make a copy of weights to register hook on it
        identity_matrix = torch.eye(self.weight.size(0), device=self.weight.device)
        weight = torch.matmul(identity_matrix, self.weight).requires_grad_(True)
        weight.register_hook(my_hook)
        return input @ weight.T


class ProposedGPTWithLogger(GPT):

    def __init__(self, config, train_steps, margin=-1, T=1):
        super().__init__(config)
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.grad_logger = GradLogger(train_steps, config.vocab_size)
        self.vocab_size = config.vocab_size

        self.transformer = nn.ModuleDict(dict(
            wte = EmbeddingWithLogger(config.vocab_size, config.n_embd, self.grad_logger),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.pos_lm_head = LinearWithLogger(config.n_embd, config.vocab_size, bias=False, grad_logger=self.grad_logger, grad_type='grad_output_positive')
        self.neg_lm_head = LinearWithLogger(config.n_embd, config.vocab_size, bias=False, grad_logger=self.grad_logger, grad_type='grad_output_negative')
        self.transformer.wte.weight = self.pos_lm_head.weight = self.neg_lm_head.weight
        self.lm_head = None

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        self.grad_logger.set_emb_prev(self.transformer.wte.weight)
        self.margin = margin
        self.T = T

    def log_output_loss(self, logits, targets):
        token2loss = torch.zeros(self.vocab_size, device=logits.device)
        token_counts = torch.zeros(self.vocab_size, device=logits.device)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction='none').detach()

        # Flatten `targets` and `loss`
        flattened_targets = targets.view(-1)
        loss = loss.view(-1)[flattened_targets != -1]
        flattened_targets = flattened_targets[flattened_targets != -1]

        # Accumulate loss and counts for each token ID
        token2loss.scatter_add_(0, flattened_targets, loss)
        token_counts.scatter_add_(0, flattened_targets, torch.ones_like(loss))

        # Avoid division by zero for tokens that don't appear in the batch
        token2loss = torch.where(token_counts > 0, token2loss / token_counts, token2loss)
        self.grad_logger.set_history("output_loss", token2loss)

    def log_grads(self, lr, weight_decay):
        grad_logger = self.grad_logger
        # log emb
        grad_logger.set_history("emb", torch.norm(self.transformer.wte.weight, dim=-1))
        # log grad_total
        grad_logger.set_history("grad_total", torch.norm(self.transformer.wte.weight.grad, dim=-1))
        emb_current = self.transformer.wte.weight.clone().detach()
        emb_prev = grad_logger.get_emb_prev()
        # lr * (grad_adam_total + adam_weight_decay) = emb_current - emb_prev
        delta_emb = emb_current - emb_prev
        grad_adam_total = delta_emb / lr
        grad_adam_decay = emb_prev * weight_decay
        grad_adam_moments = grad_adam_total - grad_adam_decay
        grad_logger.set_history("grad_adam_total", torch.norm(grad_adam_total, dim=-1))
        grad_logger.set_history("grad_adam_decay", torch.norm(grad_adam_decay, dim=-1))
        grad_logger.set_history("grad_adam_moments", torch.norm(grad_adam_moments, dim=-1))
        grad_logger.set_emb_prev(emb_current)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # logits = self.lm_head(x)

            # if we are given some desired targets also calculate the loss
            # calculate the same logits from 2 identical heads (they share weights)
            pos_logits = self.pos_lm_head(x)
            neg_logits = self.neg_lm_head(x)
            # returns onehot encoding of targets, tensor of size (targets.size[0], targets.size[1], vocab_size)
            positive_mask = torch.arange(self.vocab_size).view(1, 1, -1) == targets.unsqueeze(-1)
            # combine logits from 2 sources together
            logits = torch.where(positive_mask, pos_logits, neg_logits)
            logits *= self.T
            if self.margin != -1:
                threshold = torch.gather(logits, 2, targets.unsqueeze(-1)) - self.margin
                logits = torch.where(logits < threshold, torch.tensor(-float('Inf'), device=logits.device), logits)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            self.log_output_loss(pos_logits, targets)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.pos_lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def forward_eval(self, idx, targets):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.pos_lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.pos_lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss