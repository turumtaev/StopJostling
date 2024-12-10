import math
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPT, Block, LayerNorm

class SeparatedEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        # Create a list of separate embeddings (one for each token)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(embedding_dim)) for _ in range(num_embeddings)
        ])

    def forward(self, input):
        weight = torch.stack([w for w in self.weights]).requires_grad_(True)
        return weight[input]

class SeparatedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        # Create a list of weight vectors, one for each output feature
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(in_features)) for _ in range(out_features)
        ])
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        weight = torch.stack([w for w in self.weights]).requires_grad_(True)
        output = input @ weight.T
        
        # Add bias if necessary
        if self.bias is not None:
            output += self.bias
        return output

class ProposedGPTWithSeparatedEmbeddings(GPT):
    def __init__(
        self, config,
        baseline=False,
        weight_tying=True,
        margin=1, T=1,
        margin_by_weight=False, margin_by_weight_alpha=1.0  # to try margin depends on embedding norms
        ):
        super().__init__(config, weight_tying=weight_tying)
        self.vocab_size = config.vocab_size

        self.transformer.wte = SeparatedEmbedding(config.vocab_size, config.n_embd)
        self.lm_head = SeparatedLinear(config.n_embd, config.vocab_size, bias=False)
        if weight_tying:
            for i in range(self.vocab_size):
                self.transformer.wte.weights[i] = self.lm_head.weights[i]

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        self.margin = margin
        self.T = T

        self.margin_by_weight = margin_by_weight
        self.margin_by_weight_alpha = margin_by_weight_alpha
        self.baseline = baseline

    def _init_weights(self, module):
        if isinstance(module, SeparatedLinear) or isinstance(module, SeparatedEmbedding):
            # Initialize the weights for each individual parameter in the ParameterList
            for i in range(len(module.weights)):
                torch.nn.init.normal_(module.weights[i], mean=0.0, std=0.02)
            if isinstance(module, SeparatedLinear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits *= self.T
            if not self.baseline:
                if self.margin_by_weight:
                    if self.margin_by_weight == 1:
                        weight = torch.stack([w for w in self.lm_head.weights])
                        W_norms = torch.norm(weight, dim=-1)  # (vocab_size)
                        margin = W_norms[targets.unsqueeze(-1)] * self.margin_by_weight_alpha
                    elif self.margin_by_weight == 2:
                        margin = torch.norm(x, dim=-1).unsqueeze(-1) * self.margin_by_weight_alpha
                    elif self.margin_by_weight == 3:
                        weight = torch.stack([w for w in self.lm_head.weights])
                        W_norms = torch.norm(weight, dim=-1)  # (vocab_size)
                        margin = W_norms[targets.unsqueeze(-1)] * torch.norm(x, dim=-1).unsqueeze(-1) * self.margin_by_weight_alpha
                else:
                    margin = self.margin

                threshold = torch.gather(logits, 2, targets.unsqueeze(-1)) - margin
                logits = torch.where(logits < threshold, torch.tensor(-float('Inf'), device=logits.device), logits)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
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
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 or ('transformer.wte.weights' in n)]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 and ('transformer.wte.weights' not in n)]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer