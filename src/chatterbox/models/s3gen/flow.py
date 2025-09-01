# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from .utils.mask import make_pad_mask
from .configs import CFM_PARAMS


class MaskedDiffWithXvec(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: str = "mel",
        vocab_size: int = 4096,
        input_frame_rate: int = 50,
        only_mask_loss: bool = True,
        encoder: torch.nn.Module = None,
        length_regulator: torch.nn.Module = None,
        decoder: torch.nn.Module = None,
        decoder_conf: Dict = {
            "in_channels": 240,
            "out_channel": 80,
            "spk_emb_dim": 80,
            "n_spks": 1,
            "cfm_params": CFM_PARAMS,
            "decoder_params": {
                "channels": [256, 256],
                "dropout": 0.0,
                "attention_head_dim": 64,
                "n_blocks": 4,
                "num_mid_blocks": 12,
                "num_heads": 8,
                "act_fn": "gelu",
            },
        },
        mel_feat_conf: Dict = {
            "n_fft": 1024,
            "num_mels": 80,
            "sampling_rate": 22050,
            "hop_size": 256,
            "win_size": 1024,
            "fmin": 0,
            "fmax": 8000,
        },
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.length_regulator = length_regulator
        self.only_mask_loss = only_mask_loss

    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        token = batch["speech_token"].to(device)
        token_len = batch["speech_token_len"].to(device)
        feat = batch["speech_feat"].to(device)
        feat_len = batch["speech_feat_len"].to(device)
        embedding = batch["embedding"].to(device)

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        h, h_lengths = self.length_regulator(h, feat_len)

        # get conditions
        conds = torch.zeros(feat.shape, device=token.device)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(feat_len)).to(h)
        feat = F.interpolate(
            feat.unsqueeze(dim=1), size=h.shape[1:], mode="nearest"
        ).squeeze(dim=1)
        loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding,
            cond=conds,
        )
        return {"loss": loss}

    @torch.inference_mode()
    def inference(
        self,
        token,
        token_len,
        prompt_token,
        prompt_token_len,
        prompt_feat,
        prompt_feat_len,
        embedding,
        flow_cache,
    ):
        if self.fp16 is True:
            prompt_feat = prompt_feat.half()
            embedding = embedding.half()

        assert token.shape[0] == 1
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        token_len1, token_len2 = prompt_token.shape[1], token.shape[1]
        token, token_len = (
            torch.concat([prompt_token, token], dim=1),
            prompt_token_len + token_len,
        )
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        mel_len1, mel_len2 = prompt_feat.shape[1], int(
            token_len2 / self.input_frame_rate * 22050 / 256
        )
        h, h_lengths = self.length_regulator.inference(
            h[:, :token_len1],
            h[:, token_len1:],
            mel_len1,
            mel_len2,
            self.input_frame_rate,
        )

        # get conditions
        conds = torch.zeros(
            [1, mel_len1 + mel_len2, self.output_size], device=token.device
        ).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, flow_cache = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            prompt_len=mel_len1,
            flow_cache=flow_cache,
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), flow_cache


class CausalMaskedDiffWithXvec(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: str = "mel",
        vocab_size: int = 6561,
        input_frame_rate: int = 25,
        only_mask_loss: bool = True,
        token_mel_ratio: int = 2,
        pre_lookahead_len: int = 3,
        encoder: torch.nn.Module = None,
        decoder: torch.nn.Module = None,
        decoder_conf: Dict = {
            "in_channels": 240,
            "out_channel": 80,
            "spk_emb_dim": 80,
            "n_spks": 1,
            "cfm_params": CFM_PARAMS,
            "decoder_params": {
                "channels": [256, 256],
                "dropout": 0.0,
                "attention_head_dim": 64,
                "n_blocks": 4,
                "num_mid_blocks": 12,
                "num_heads": 8,
                "act_fn": "gelu",
            },
        },
        mel_feat_conf: Dict = {
            "n_fft": 1024,
            "num_mels": 80,
            "sampling_rate": 22050,
            "hop_size": 256,
            "win_size": 1024,
            "fmin": 0,
            "fmax": 8000,
        },
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len

        # FIXME: this was missing - just putting it in as false
        self.fp16 = False

    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Training forward for the causal masked-diff model.

        Behavior:
        - embed tokens and speaker xvector
        - encode tokens (encoder expected to produce mel-aligned frames)
        - build prefix conditions (random short prefixes of the target mel)
        - interpolate feat/conds to match encoder output length if necessary
        - compute loss via decoder.compute_loss (same signature as MaskedDiffWithXvec)
        """
        token = batch["speech_token"].to(device)
        token_len = batch["speech_token_len"].to(device)
        feat = batch["speech_feat"].to(device)  # shape [B, T_feat, C]
        feat_len = batch["speech_feat_len"].to(device)
        embedding = batch["embedding"].to(device)

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # token embedding (mask paddings)
        token_mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        token = self.input_embedding(torch.clamp(token, min=0)) * token_mask

        # encode (encoder here is expected to produce mel-aligned frames)
        h, h_lengths = self.encoder(token, token_len)  # h: [B, T_enc, C_enc]
        h = self.encoder_proj(h)  # project to output_size -> [B, T_enc, output_size]

        # ----- prepare prefix conditions -----
        # conds starts with original feat shape (B, T_feat, C)
        conds = torch.zeros_like(feat, device=feat.device)

        # fill in random prefixes (up to 30% of the true length) for some examples
        for i, true_len in enumerate(feat_len):
            # true_len is a tensor scalar; convert to int safely
            j = int(true_len.item())
            if j <= 0:
                continue
            if random.random() < 0.5:
                continue
            idx = random.randint(0, max(0, int(0.3 * j)))
            if idx > 0:
                conds[i, :idx] = feat[i, :idx]

        # Now we will align feat and conds to the encoder length (T_enc).
        # This makes the method robust in case the encoder output length differs
        # from the raw feature frame count.
        T_enc = h.shape[1]
        # Interpolate feat to encoder time dimension (and channel dimension if needed).
        feat = F.interpolate(
            feat.unsqueeze(1), size=(T_enc, feat.shape[2]), mode="nearest"
        ).squeeze(1)
        # Interpolate conds as well so conds aligns with h (will be zeros where we didn't copy).
        conds = F.interpolate(
            conds.unsqueeze(1), size=(T_enc, conds.shape[2]), mode="nearest"
        ).squeeze(1)
        # transpose conds to [B, C, T] for decoder API
        conds = conds.transpose(1, 2)

        # Build a time-mask aligned to encoder length.
        # Map original feat_len (which indexes into original max-feat-length) to encoder length
        orig_feat_max_len = (
            batch["speech_feat"].shape[1]
            if isinstance(batch.get("speech_feat"), torch.Tensor)
            else feat.shape[1]
        )
        # Avoid division by zero; if orig_feat_max_len equals 0 (shouldn't happen), fall back to T_enc
        if orig_feat_max_len == 0:
            scaled_feat_len = torch.tensor(
                [T_enc] * feat.shape[0], device=device, dtype=torch.long
            )
        else:
            # scale each sample's length proportionally
            scaled = (
                (feat_len.float() * (T_enc / float(orig_feat_max_len))).round().long()
            )
            # clamp to valid range [0, T_enc]
            scaled_feat_len = torch.clamp(scaled, min=0, max=T_enc).to(device)

        mask = (~make_pad_mask(scaled_feat_len)).to(h)

        # prepare inputs for decoder.compute_loss: shapes expected are
        # feat.transpose(1,2): [B, C, T]
        # mask.unsqueeze(1): [B, 1, T]
        # h.transpose(1,2): [B, C, T]
        loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding,
            cond=conds,
        )

        return {"loss": loss}

    @torch.inference_mode()
    def inference(
        self,
        token,
        token_len,
        prompt_token,
        prompt_token_len,
        prompt_feat,
        prompt_feat_len,
        embedding,
        finalize,
    ):
        if self.fp16 is True:
            prompt_feat = prompt_feat.half()
            embedding = embedding.half()

        assert token.shape[0] == 1
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        token, token_len = (
            torch.concat([prompt_token, token], dim=1),
            prompt_token_len + token_len,
        )
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        if finalize is False:
            h = h[:, : -self.pre_lookahead_len * self.token_mel_ratio]
        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]
        h = self.encoder_proj(h)

        # get conditions
        conds = torch.zeros(
            [1, mel_len1 + mel_len2, self.output_size], device=token.device
        ).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), None  # NOTE jrm: why are they returning None here?
