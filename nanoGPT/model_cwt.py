import torch
from torch.nn import functional as F
from model import GPT

class CWTGPT(GPT):

    def __init__(self, config):
        super().__init__(config)

    def cwt_loss(self, input_embs, target_embs):
        # input_embs: nb_embs x hidden_dim
        # target_embs: nb_embs x hidden_dim

        exp_cosine_sim = torch.exp(torch.mm(input_embs, target_embs.T))
        self_dist = exp_cosine_sim.diagonal()
        neg_dist = exp_cosine_sim.sum(-1)

        return -(self_dist/(neg_dist + 1e-9)).log().mean()


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
            # # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

            target_embs = self.transformer.wte(targets)
            loss = self.cwt_loss(x.view(-1, x.size(-1)), target_embs.view(-1, target_embs.size(-1)))
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss