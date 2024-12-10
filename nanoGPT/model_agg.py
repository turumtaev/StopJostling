import torch
from torch.nn import functional as F
from model import GPT

class AGGGPT(GPT):

    def __init__(self, config, alpha=0.02, K=1600):
        super().__init__(config)
        self.alpha = alpha
        self.is_rare = None
        self.alpha = alpha
        self.count_memory = K
        self.word_count = torch.zeros(self.count_memory+1, config.vocab_size)
        self.itr = 0

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
            self.itr += 1
            loss = self.compute_agg(x, targets, itr=self.itr)

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


    def compute_agg(self, x, targets, itr=1):
        alpha = self.alpha

        features = x
        target = targets.view(-1)

        count_mem_idx = itr % (self.count_memory + 1)
        self.word_count[count_mem_idx] = 0
        self.word_count[count_mem_idx - 1, target] += (target.unsqueeze(1) == target).float().sum(1)

        # get token appearance
        word_count = self.word_count.sum(0)

        # grouping rare tokens according to the token appearance in the current step
        f1 = word_count / min(itr, self.count_memory)
        rare_group_mask = (f1 < alpha)
        rare_group_indices = rare_group_mask.float().nonzero().squeeze(-1)
        non_rare_group_indices = (1 - rare_group_mask.float()).nonzero().squeeze(-1)
        f1[non_rare_group_indices] = 1.

        # calculate normalized frequency of very-rare tokens
        c_r_mean = word_count[rare_group_indices].mean()
        f2 = word_count / (c_r_mean + 1e-7)  # abs_p_rare
        f2[non_rare_group_indices] = 1.
        f2 = f2.clip(max=1.)

        output_weight = self.lm_head.weight * 2 / 2
        target_mask = (target.unsqueeze(1) == rare_group_indices).float().sum(1)
        rare_tgt_idx = target_mask.nonzero().squeeze(-1)
        non_rare_tgt_idx = (1. - target_mask).nonzero().squeeze(-1)

        # calculate gate1 and gate2 for gating gradients of the token embedding vectors
        gate1 = f1.expand(non_rare_tgt_idx.size(0), output_weight.size(0)).clone().detach()
        gate1[torch.arange(non_rare_tgt_idx.size(0)), target[non_rare_tgt_idx]] = 1.
        gate2 = f2.expand(rare_tgt_idx.size(0), output_weight.size(0)).clone().detach()
        gate2[torch.arange(rare_tgt_idx.size(0)), target[rare_tgt_idx]] = 1.

        # calculate the gated logits for gate1
        features_size = features.size()
        features_nonrare = features.detach().contiguous().view(-1, features_size[2])[non_rare_tgt_idx]
        logits_nonrare = torch.nn.functional.linear(features_nonrare, output_weight)
        logits_size = logits_nonrare.size()
        logits_nonrare = logits_nonrare.view(-1, logits_size[-1])
        logits_nonrare_gated = gate1 * logits_nonrare + (1.-gate1) * logits_nonrare.detach()

        # calculate the gated logits for gate2
        features_rare = features.detach().contiguous().view(-1, features_size[2])[rare_tgt_idx]
        logits_rare = torch.nn.functional.linear(features_rare, output_weight)
        logits_rare = logits_rare.view(-1, logits_size[-1])
        logits_rare_gated = gate2 * logits_rare + (1.-gate2) * logits_rare.detach()

        # calculate the original nll logits for gradients about feature vectors
        logits_feature = torch.nn.functional.linear(features, output_weight.detach())
        logits_feature = logits_feature.view(-1, logits_size[-1])

        loss_1 = F.cross_entropy(logits_feature.float().view(-1, logits_feature.size(1)), target, ignore_index=-1, reduction='sum')
        loss_2 = F.cross_entropy(logits_nonrare_gated.float().view(-1, logits_nonrare_gated.size(1)), target[non_rare_tgt_idx], ignore_index=-1, reduction='sum')
        loss_3 = F.cross_entropy(logits_rare_gated.float().view(-1, logits_rare_gated.size(1)), target[rare_tgt_idx], ignore_index=-1, reduction='sum')
        loss = (loss_1 + loss_2 + loss_3) / len(target)

        return loss