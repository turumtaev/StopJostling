import torch
from torch.nn import functional as F
from model import GPT

def softminus_thresholding(logits, threshold, c=1):
    mask = logits > threshold
    new_logits = torch.full_like(logits, float('-inf'))  # Initialize the result tensor with -inf
    expanded_threshold = threshold.expand_as(logits)
    # new_logits[mask] = c * torch.log(torch.exp((logits[mask] - expanded_threshold[mask])/c) - 1) + expanded_threshold[mask]
    new_logits[mask] = c * torch.log(torch.maximum(torch.tensor(1e-9, dtype=logits.dtype, device=logits.device), 
                                            torch.exp((logits[mask] - expanded_threshold[mask]) / c) - 1)) + expanded_threshold[mask]
    return new_logits

class ProposedGPT(GPT):

    def __init__(
        self,
        config,
        weight_tying=True,
        margin=1,
        T=1,  # to optimize P_{\theta, 1/T}
        softminus=False, softminus_c=1, detach_threshold=False,  # to try softminus
        detach_logits_under_threshold=False,  # to not eliminate logits under threshold, but only detach
        margin_by_weight=False, margin_by_weight_alpha=1.0  # to try margin depends on embedding norms
    ):
        super().__init__(config, weight_tying=weight_tying)
        self.margin = margin
        self.T = T
        self.softminus = softminus
        self.softminus_c = softminus_c
        self.detach_threshold = detach_threshold
        self.detach_logits_under_threshold = detach_logits_under_threshold
        assert not detach_logits_under_threshold or not softminus, "softminus doesn't support detach_logits_under_threshold"
        self.margin_by_weight = margin_by_weight
        self.margin_by_weight_alpha = margin_by_weight_alpha

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
            if self.margin_by_weight:
                if self.margin_by_weight == 1:
                    W_norms = torch.norm(self.lm_head.weight, dim=-1)  # (vocab_size)
                    margin = W_norms[targets.unsqueeze(-1)] * self.margin_by_weight_alpha
                elif self.margin_by_weight == 2:
                    margin = torch.norm(x, dim=-1).unsqueeze(-1) * self.margin_by_weight_alpha
                elif self.margin_by_weight == 3:
                    W_norms = torch.norm(self.lm_head.weight, dim=-1)  # (vocab_size)
                    margin = W_norms[targets.unsqueeze(-1)] * torch.norm(x, dim=-1).unsqueeze(-1) * self.margin_by_weight_alpha
            else:
                margin = self.margin

            threshold = torch.gather(logits, 2, targets.unsqueeze(-1)) - margin
            if self.detach_threshold:
                threshold = threshold.detach()

            if self.softminus:
                logits = softminus_thresholding(logits, threshold, self.softminus_c)
            elif self.detach_logits_under_threshold:
                logits_detached = (1 * logits).detach()
                logits = torch.where(logits < threshold, logits_detached, logits)
            else:
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