import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from torchinfo import summary
from chatterbox.models.t3.modules.t3_config import T3Config
from dataset_utils import SimpleMetaDataset, collate_t3
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from safetensors.torch import load_file
from chatterbox.models.t3.t3 import T3
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.tokenizers import EnTokenizer

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    en_tok = EnTokenizer(str(args.tokenizer))

    ds = SimpleMetaDataset(
        args.meta,
        cache_dir=args.cache_dir,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_t3(b, en_tok),
    )
    t3_conf = T3Config()
    t3 = T3(t3_conf)
    state = load_file(Path(args.ckpt_dir) / "t3_cfg.safetensors")
    t3.load_state_dict(state)

    tok_vocab = len(en_tok.tokenizer.get_vocab())

    cur_size = getattr(t3.text_emb, "num_embeddings", t3.text_emb.weight.size(0))

    if tok_vocab > cur_size:
        old_emb = t3.text_emb
        new_emb = torch.nn.Embedding(tok_vocab, old_emb.embedding_dim)
        with torch.no_grad():
            new_emb.weight[:cur_size].copy_(old_emb.weight[:cur_size])
            std = float(old_emb.weight.std().item())
            torch.nn.init.normal_(new_emb.weight[cur_size:], mean=0.0, std=std)
        t3.text_emb = new_emb

        if hasattr(t3, "text_head"):
            old_head = t3.text_head
            hidden = old_head.in_features
            new_head = torch.nn.Linear(
                hidden, tok_vocab, bias=(old_head.bias is not None)
            )
            with torch.no_grad():
                ncopy = min(old_head.out_features, tok_vocab)
                new_head.weight[:ncopy].copy_(old_head.weight[:ncopy])
                if tok_vocab > ncopy:
                    std = float(old_head.weight.std().item())
                    torch.nn.init.normal_(new_head.weight[ncopy:], mean=0.0, std=std)
                if old_head.bias is not None:
                    new_head.bias[:ncopy].copy_(old_head.bias[:ncopy])
            t3.text_head = new_head

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "donw_proj",
        ],
        inference_mode=False,
    )

    t3.to("cpu")
    model = get_peft_model(t3, lora_config)

    base_wrapper = getattr(model, "base_model", model)
    for layer_name in ("text_emb", "text_head"):
        layer = getattr(base_wrapper, layer_name, None)
        if layer is not None:
            for pname, p in layer.named_parameters():
                if not p.requires_grad:
                    p.requires_grad_(True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if not trainable_params:
        raise RuntimeError("No trainable parameters available for training")

    summary(model)

    optim = torch.optim.Adam(trainable_params, lr=args.lr)
    tb_writer = SummaryWriter(log_dir=str(Path(args.out_dir) / "tb"))
    global_step = 0
    model.train()
    model.to(device)

    accum_ls = 0.0
    accum_lt = 0.0
    gamma = 0.2

    for epoch in range(args.epochs):
        loop = tqdm(dl, desc=f"epoch {epoch}")

        for i, batch in enumerate(loop):
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }

            txt = batch["text_tokens"].to(device)
            txt_lens = batch["text_lens"].to(device)
            speech_tokens = batch["speech_tokens"].to(device)
            speech_lens = batch["speech_lens"].to(device)
            speaker_embs = batch["speaker_embs"].to(device)

            B = speaker_embs.size(0)
            emotion_tensor = torch.full((B, 1, 1), float(0.5), device=device)
            t3_cond = T3Cond(speaker_emb=speaker_embs, emotion_adv=emotion_tensor)

            lt, ls = model.loss(
                t3_cond=t3_cond,
                text_tokens=txt,
                text_token_lens=txt_lens,
                speech_tokens=speech_tokens,
                speech_token_lens=speech_lens,
            )

            # combined loss for backward
            loss = (ls * gamma + lt * (1 - gamma)) / args.grad_accum_steps
            accum_ls += ls.detach().cpu().item() / args.grad_accum_steps
            accum_lt += lt.detach().cpu().item() / args.grad_accum_steps
            loss.backward()
            if i % args.grad_accum_steps == 0:
                optim.step()
                optim.zero_grad()
                loop.set_postfix(
                    loss=accum_ls * gamma + accum_lt * (1 - gamma),
                    loss_text=accum_lt,
                    loss_speech=accum_ls,
                )
                tb_writer.add_scalar(
                    "train/loss",
                    accum_ls * gamma + accum_lt * (1 - gamma),
                    global_step,
                )
                tb_writer.add_scalar(
                    "train/loss_text",
                    accum_lt,
                    global_step,
                )
                tb_writer.add_scalar(
                    "train/loss_speech",
                    accum_ls,
                    global_step,
                )
                global_step += 1
                accum_lt = 0.0
                accum_ls = 0.0

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    peft_state = get_peft_model_state_dict(model)
    torch.save(peft_state, Path(args.out_dir) / "lora_t3_state.pt")
    base_updates = {
        "text_emb": t3.text_emb.state_dict(),
        "text_head": t3.text_head.state_dict(),
    }
    torch.save(base_updates, Path(args.out_dir) / "base_updates.pt")


if __name__ == "__main__":
    main()
