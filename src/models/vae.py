

import pytorch_lightning as pl
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random
from datasets import MidiDataModule
from vocab import RemiVocab
from constants import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN

import transformers
from transformers import (
  BertConfig,
  EncoderDecoderConfig,
  EncoderDecoderModel
)


# Implementation adapted from https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py
# Random restarts adapted from https://github.com/openai/jukebox/blob/master/jukebox/vqvae/bottleneck.py
class VectorQuantizeEMA(nn.Module):
    def __init__(self, d_latent, n_codes, n_groups=1, decay=0.995, eps=1e-4, restart_threshold=0.99):
        assert d_latent // n_groups == d_latent / n_groups, f"Unexpected latent dimension: d_latent={d_latent} must be divisible by n_groups={n_groups}"

        super().__init__()

        self.d_latent = d_latent
        self.n_groups = n_groups
        self.dim = d_latent // n_groups
        self.n_codes = n_codes

        self.decay = decay
        self.eps = eps
        self.threshold = restart_threshold
        self.init = False

        embed = torch.randn(self.n_codes, self.dim)
        self.register_buffer('embedding', embed)
        self.register_buffer('cluster_size', torch.ones(self.n_codes))
        self.register_buffer('cluster_sum', embed.clone().detach())

    def forward(self, x, dist=None):
        assert x.shape[-1] == self.n_groups * self.dim, f"Unexpected input shape: expected last dimension to be {self.n_groups * self.dim} but was {x.shape[-1]}"
        x_ = x.reshape(-1, self.dim)

        if self.training and not self.init:
            self._init_embeddings(x_, dist=dist)

        ### Shared embeddings between groups ###
        # Find nearest neighbors in latent space
        emb_t = self.embedding.t()
        distance = (
            x_.pow(2).sum(1, keepdim=True)
            - 2 * x_ @ emb_t
            + emb_t.pow(2).sum(0, keepdim=True)
        )
        _, embed_idx = (-distance).max(1)
        embed_onehot = F.one_hot(embed_idx, self.n_codes).type(x_.dtype)

        quantize = self.embed(embed_idx).view(-1, self.n_groups * self.dim)
        diff = (quantize.detach() - x).pow(2).mean()
        quantize = x + (quantize - x).detach()
        codes = embed_idx.view(-1, self.n_groups)

        if self.training:
            update_metrics = self._ema_update(x_, embed_onehot, dist=dist)
        else:
            update_metrics = {}

        return dict(
            z=quantize,
            diff=diff,
            codes=codes,
            **update_metrics
        )

    def embed(self, idx):
        return F.embedding(idx, self.embedding)

    def _init_embeddings(self, x, dist=None):
        self.init = True
        rand_centers = self._randomize(x)
        self.cluster_sum.data.copy_(rand_centers)
        self.cluster_size.data.fill_(1)
        

    def _randomize(self, x):
        n = x.size(0)
        if n < self.n_codes:
            r = (self.n_codes + n - 1) // n  # r = math.ceil(n_codes / n)
            std = 0.01 / np.sqrt(self.dim)
            x = x.repeat(r, 1)
            x += std * torch.randn_like(x)
        return x[torch.randperm(x.size(0))][:self.n_codes]

    def _ema_update(self, x, cluster_assign, dist=None):
        with torch.no_grad():
            cluster_size = cluster_assign.sum(0)
            cluster_sum = cluster_assign.t() @ x

            rand_centers = self._randomize(x)

            # Gather results from all GPUs to get better estimate
            # This doesn't work for the DataParallel accelerator
            # if dist is not None:
            #     dist.broadcast(rand_centers)
            #     cluster_size = dist.reduce(cluster_size, reduce_op='sum')
            #     cluster_sum = dist.reduce(cluster_sum, reduce_op='sum')

            # EMA update step
            # self.cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1-self.decay)
            # self.cluster_sum.data.mul_(self.decay).add_(cluster_sum, alpha=1-self.decay)
            self.cluster_size.data.copy_(self.decay*self.cluster_size + (1 - self.decay)*cluster_size)
            self.cluster_sum.data.copy_(self.decay*self.cluster_sum + (1 - self.decay)*cluster_sum)

            used = (self.cluster_size >= self.threshold).float().unsqueeze(-1)

            n = self.cluster_size.sum()
            # Use additive smoothing to mitigate exploding gradients
            count = (self.cluster_size + self.eps) / (n + self.n_codes*self.eps) * n

            cluster_centers = self.cluster_sum / count.unsqueeze(-1)
            cluster_centers = used * cluster_centers + (1 - used) * rand_centers
            self.embedding.data.copy_(cluster_centers)

            # Also reset size of cluster when doing random restarts => prevent from randomly restarting many times in a row
            # new_sizes = used.squeeze(1) * self.cluster_size + (1 - used.squeeze(1))
            # self.cluster_size.data.copy_(new_sizes)

            # Compute metrics
            avg_usage = used.mean()
            usage = used.sum()
            pr = cluster_size / cluster_size.sum()
            entropy = -(pr * (pr + 1e-5).log()).sum()

        return {
            'avg_usage': avg_usage,
            'usage': usage,
            'entropy': entropy
        }


class VqVaeModule(pl.LightningModule):
    def __init__(self, 
                 d_model=512,
                 context_size=256, 
                 n_codes=1024, 
                 n_groups=2,
                 d_latent=1024,
                 lr=1e-4, 
                 lr_schedule='sqrt_decay', 
                 warmup_steps=1000, 
                 max_steps=10000,
                 encoder_layers=6,
                 decoder_layers=6,
                 encoder_ffn_dim=2048,
                 decoder_ffn_dim=2048,
                 windowed_attention_pr=0.0,
                 max_lookahead=4,
                 disable_vq=False):
        super().__init__()

        self.d_model = d_model
        self.context_size = context_size
        self.n_codes = n_codes
        self.n_groups = n_groups
        self.d_latent = d_latent

        self.beta = 0.02
        self.cycle_length = 2000

        self.lr = lr
        self.lr_schedule = lr_schedule
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.windowed_attention_pr = windowed_attention_pr
        self.max_lookahead = max_lookahead
        self.disable_vq = disable_vq

        self.vocab = RemiVocab()
        
        self.pad_token = self.vocab.to_i(PAD_TOKEN)
        self.bos_token = self.vocab.to_i(BOS_TOKEN)
        self.eos_token = self.vocab.to_i(EOS_TOKEN)
        self.mask_token = self.vocab.to_i(MASK_TOKEN)

        encoder_config = BertConfig(
            vocab_size=1,
            pad_token_id=0,
            hidden_size=self.d_model,
            num_hidden_layers=encoder_layers,
            num_attention_heads=8,
            intermediate_size=encoder_ffn_dim,
            max_position_embeddings=1024,
            position_embedding_type='relative_key_query'
        )
        decoder_config = BertConfig(
            vocab_size=1,
            pad_token_id=0,
            hidden_size=self.d_model,
            num_hidden_layers=decoder_layers,
            num_attention_heads=8,
            intermediate_size=decoder_ffn_dim,
            max_position_embeddings=1024,
            position_embedding_type='relative_key_query'
        )
        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
        self.transformer = EncoderDecoderModel(config)
        self.transformer.config.decoder.is_decoder = True
        self.transformer.config.decoder.add_cross_attention = True
        self.encoder = self.transformer.encoder
        self.decoder = self.transformer.decoder

        self.in_layer = nn.Embedding(len(self.vocab), self.d_model)
        self.out_layer = nn.Linear(self.d_model, len(self.vocab), bias=False)
        
        self.vq_embed = VectorQuantizeEMA(self.d_latent, self.n_codes, self.n_groups)
        self.pooling = nn.Linear(self.d_model, self.d_latent, bias=False)
        self.unpooling = nn.Linear(self.d_latent, self.d_model, bias=False)
        self.attention_proj = nn.Linear(self.d_model, self.d_model)

        self.rec_loss = nn.CrossEntropyLoss(ignore_index=self.pad_token)

        self.save_hyperparameters()
    
    def get_datamodule(self, midi_files, **kwargs):
        return MidiDataModule(
            midi_files, 
            self.context_size, 
            max_bars_per_context=1, 
            bar_token_mask=MASK_TOKEN,
            **kwargs
        )

    def forward(self, x, y=None, latent=None, use_windowed_attention=False, return_latent_logits=False):
        if y is None:
            y = x.clone().detach()

        # VQ-VAE
        if latent is None:
            encoder_out = self.encode(x)
            latent = encoder_out['z']
        
        logits = self.decode(x, latent, use_windowed_attention)
        return {
            'logits': logits,
            **encoder_out
        }
            

    def embed(self, x):
        return self.in_layer(x)

    def encode(self, x):
        x_emb = self.embed(x)

        # Shape of out: (batch_size, seq_len, d_model)
        out = self.encoder(inputs_embeds=x_emb, output_hidden_states=True)
        hidden = out.pooler_output
        # Shape of z_e: (batch_size, d_model * n_groups)
        z_e = self.pooling(hidden)

        if self.disable_vq:
            # AE baseline
            return { 'z': z_e }
        else:
            # VQ-VAE
            # Shape of z_q: (batch_size, d_model * n_groups)
            dist = self.trainer.accelerator.training_type_plugin if self.training else None
            return self.vq_embed(z_e, dist=dist)


    def decode(self, x, latent, use_windowed_attention=False):
        # Shape of latent: (batch_size, n_groups, d_model)
        x_emb = self.embed(x)
        seq_len = x_emb.size(1)

        # Shape of h0: (batch_size, d_model)
        h0 = self.unpooling(latent)

        # Make model decoder-only by fixing h0
        # h0 = torch.zeros_like(h0)

        # Strategy 1: Add latent embeddings to input embeddings
        x_emb += h0.unsqueeze(1).repeat(1, seq_len, 1)

        # Strategy 2: Use latent embedding in cross-attention
        x_attention = self.attention_proj(h0.unsqueeze(1).repeat(1, self.context_size, 1))

        # Relative pos. embeddings need source and target to be of the same length
        # -> prevents einsum shape mismatch error
        padding = torch.zeros_like(x_attention)
        padding[:, :x_emb.size(1)] = x_emb
        x_emb = padding


        if self.training or use_windowed_attention:
            attention_mask = self.rand_attention_mask(x)
        else:
            attention_mask = self.get_attention_mask(x)
        padding = torch.zeros((x.size(0), self.context_size, self.context_size), device=self.device, dtype=torch.int)
        padding[:, :attention_mask.size(1), :attention_mask.size(2)] = attention_mask
        attention_mask = padding

        out = self.decoder(
            inputs_embeds=x_emb, 
            encoder_hidden_states=x_attention, 
            attention_mask=attention_mask, 
            output_hidden_states=True
        )
        hidden = out.hidden_states[-1][:, :seq_len]
        logits = self.out_layer(hidden).contiguous()
        
        return logits

    def get_loss(self, batch, windowed_attention_pr=None):
        if windowed_attention_pr is None:
            windowed_attention_pr = self.windowed_attention_pr
        use_windowed_attention = True if random.random() < windowed_attention_pr else False

        x = batch['input_ids']
        labels = batch['labels']

        out = self.forward(
            x,
            y=labels,
            use_windowed_attention=use_windowed_attention,
        )

        logits = out['logits']
        # Reshape logits to: (batch_size * seq_len, vocab_size)
        logits = logits.view(-1, logits.size(-1))
        # Reshape labels to: (batch_size * seq_len)
        labels = labels.view(-1)

        rec_loss = self.rec_loss(logits, labels)

        if self.disable_vq:
            loss = rec_loss
        else:
            diff = out['diff']
            loss = rec_loss + self.beta*diff
        
        return {
            'loss': loss,
            'rec_loss': rec_loss,
            **out
        }
    
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        metrics = self.get_loss(batch)
        log_metrics = { key: metrics[key].detach() for key in ['loss', 'rec_loss', 'diff', 'avg_usage', 'usage', 'entropy'] if key in metrics }
        self.log('train', log_metrics, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        return metrics['loss']
    
    def validation_step(self, batch, batch_idx):
        metrics = self.get_loss(batch)
        log_metrics = { key: metrics[key].detach() for key in ['rec_loss', 'diff', 'avg_usage', 'usage', 'entropy'] if key in metrics }

        # Compute perplexity
        x, y = batch['input_ids'], batch['labels']
        pad_token_id = self.vocab.to_i(PAD_TOKEN)
        logits = metrics['logits']
        log_pr = logits.log_softmax(dim=-1)
        log_pr[y == pad_token_id] = 0 # log(pr) = log(1) for padding
        log_pr = torch.gather(log_pr, -1, y.unsqueeze(-1)).squeeze(-1)
        t = (y != pad_token_id).sum(dim=-1)
        ppl = (-log_pr.sum(dim=1) / t).exp().mean()
        log_metrics['ppl'] = ppl.detach()

        self.log('valid', log_metrics, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        # Log loss separately for model checkpoint monitor
        self.log('valid_loss', metrics['loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        return metrics['loss']
    
    def test_step(self, batch, batch_idx):
        metrics = self.get_loss(batch)
        return metrics['loss']

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        step = self.trainer.global_step

        # # beta is increased for C*R steps and then held constant for C*(1-R) steps
        # if step/2 >= self.cycle_length:
        #     self.cycle_length *= 2
        # C = self.cycle_length # cycle length
        # R = 1000 # restart steps
        # b_min, b_max = 0.0, 0.1
        # t = max(0, min(1, (step % C) / R))
        # self.beta = b_min*(1 - t) + b_max*t
        # self.log('beta', self.beta, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

    def configure_optimizers(self):
        # set LR to 1, scale with LambdaLR scheduler
        optimizer = transformers.AdamW(self.parameters(), lr=1, weight_decay=0.01)

        if self.lr_schedule == 'sqrt_decay':
            # constant warmup, then 1/sqrt(n) decay starting from the initial LR
            lr_func = lambda step: min(self.lr, self.lr / math.sqrt(max(step, 1)/self.warmup_steps))
        elif self.lr_schedule == 'linear':
            # linear warmup, linear decay
            lr_func = lambda step: min(self.lr, self.lr*step/self.warmup_steps, self.lr*(1 - (step - self.warmup_steps)/self.max_steps))
        elif self.lr_schedule == 'cosine':
            # linear warmup, cosine decay to 10% of initial LR
            lr_func = lambda step: self.lr * min(step/self.warmup_steps, 0.55 + 0.45*math.cos(math.pi*(min(step, self.max_steps) - self.warmup_steps)/(self.max_steps - self.warmup_steps)))
        else:
            # Use no lr scheduling
            lr_func = lambda step: self.lr
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
        return [optimizer], [{
            'scheduler': scheduler,
            'interval': 'step',
        }]

    def rand_attention_mask(self, x, pr=0.2, max_size=None):
        if max_size == None:
            max_size = self.max_lookahead
        if max_size is not None and self.training and random.random() < pr:
            mask_size, k = random.randint(1, max_size), 0
        else:
            mask_size, k = 1, 1
        return self.get_attention_mask(x, mask_size=mask_size, k=k)
    
    def get_attention_mask(self, x, mask_size=1, k=1):
        batch_size, seq_len = x.shape[:2]

        # Standard self-attention mask for auto-regressive modelling
        tri_mask = torch.ones((seq_len//mask_size+1, seq_len//mask_size+1), device=self.device, dtype=torch.int)
        tri_mask = torch.triu(tri_mask, diagonal=k)
        tri_mask = (~tri_mask.bool()).int()
        # Create windowed self-attention mask, forcing the model to prefict farther into the future
        window_mask = tri_mask.repeat_interleave(mask_size, dim=0).repeat_interleave(mask_size, dim=1)[:seq_len, :seq_len]
        # First token needs to be always visible
        window_mask[:, 0] = 1

        return window_mask.unsqueeze(0).repeat(batch_size, 1, 1)



