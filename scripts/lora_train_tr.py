import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from dataset_utils import SimpleMetaDataset, recreate_missing_cache
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from safetensors.torch import load_file
from chatterbox.models.s3gen import S3Gen
from chatterbox.models.s3tokenizer import S3Tokenizer
from chatterbox.models.t3.t3 import T3
from chatterbox.models.tokenizers import EnTokenizer

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)


def load_t3_from_ckpt(t3, ckpt_dir):
    try:
        from safetensors.torch import load_file
    except Exception:
        return
    try:
        state = load_file(Path(ckpt_dir) / "t3_cfg.safetensors")
    except Exception:
        return
    try:
        if "model" in state:
            state = state["model"][0]
    except Exception:
        pass
    import torch as _torch

    for k, v in list(state.items()):
        try:
            tensor = _torch.as_tensor(v)
        except Exception:
            continue
        parts = k.split(".")
        obj = t3
        ok = True
        for p in parts[:-1]:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False
                break
        if not ok:
            continue
        name = parts[-1]
        if not hasattr(obj, name):
            continue
        param = getattr(obj, name)
        try:
            if param.shape == tensor.shape:
                param.copy_(tensor)
        except Exception:
            pass


def collate_t3(batch, s3tok, en_tok):
    import torchaudio as ta

    texts = []
    speech_tokens = []
    speech_lens = []

    for it in batch:
        st = it.get("speech_tokens")
        sl = it.get("speech_lens")
        if st is not None:
            if not torch.is_tensor(st):
                st = torch.as_tensor(st, dtype=torch.long)
            if sl is None:
                sl = torch.tensor(st.size(0), dtype=torch.long)
            else:
                if not torch.is_tensor(sl):
                    sl = torch.as_tensor(sl, dtype=torch.long)
        else:
            wav_path = it.get("wav")
            if wav_path is None:
                waveform = torch.zeros(1, 16000)
                sr = 16000
            else:
                try:
                    waveform, sr = ta.load(wav_path)
                except Exception:
                    waveform = torch.zeros(1, 16000)
                    sr = 16000
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != 16000:
                waveform = ta.transforms.Resample(sr, 16000)(waveform)
            wav_tensor = waveform.squeeze(0)
            toks, lens = s3tok([wav_tensor])
            st = toks[0]
            sl = lens[0]
            if not torch.is_tensor(st):
                st = torch.as_tensor(st, dtype=torch.long)
            if not torch.is_tensor(sl):
                sl = torch.as_tensor(sl, dtype=torch.long)
        speech_tokens.append(st)
        speech_lens.append(
            int(
                sl.tolist()
                if torch.is_tensor(sl) and sl.numel() > 1
                else (sl.item() if torch.is_tensor(sl) else int(sl))
            )
        )
        texts.append(it.get("text", ""))

    padded_speech = torch.nn.utils.rnn.pad_sequence(
        speech_tokens, batch_first=True, padding_value=0
    ).long()
    speech_lens_tensor = torch.tensor([int(x) for x in speech_lens], dtype=torch.long)

    sot_id = en_tok.tokenizer.token_to_id("[START]")
    eot_id = en_tok.tokenizer.token_to_id("[STOP]")
    txt_ids = []
    for t_ids, txt in zip([None] * len(texts), texts):
        if t_ids is not None:
            ids = [sot_id] + list(t_ids) + [eot_id]
        else:
            ids = [sot_id] + en_tok.encode(txt) + [eot_id]
        txt_ids.append(torch.tensor(ids, dtype=torch.long))
    txt_padded = torch.nn.utils.rnn.pad_sequence(
        txt_ids, batch_first=True, padding_value=0
    )
    txt_lens = torch.tensor([t.numel() for t in txt_ids], dtype=torch.long)

    out = {
        "text_tokens": txt_padded,
        "text_lens": txt_lens,
        "speech_tokens": padded_speech,
        "speech_lens": speech_lens_tensor,
    }
    return out


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
    p.add_argument("--force_recompute", action="store_true")
    p.add_argument("--auto_offload", action="store_true")
    p.add_argument("--single_device_transformer", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument(
        "--fast",
        action="store_true",
        help="Enable bnb 8-bit optimizer and bfloat16/autocast fast path (used in s3gen script)",
    )
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    en_tok = EnTokenizer(str(args.tokenizer))
    s3tok = S3Tokenizer()
    try:
        if torch.cuda.is_available() and args.num_workers == 0:
            s3tok.to("cuda")
    except Exception:
        pass

    ds = SimpleMetaDataset(
        args.meta,
        s3tok,
        sample_rate=16000,
        cache_dir=args.cache_dir,
        force_recompute=args.force_recompute,
    )

    s3gen = S3Gen()
    state = load_file(Path(args.ckpt_dir) / "s3gen.safetensors")
    res = s3gen.load_state_dict(state, strict=False)
    if getattr(res, "missing_keys", None):
        logger.warning("Missing keys in checkpoint: %s", res.missing_keys)
    if getattr(res, "unexpected_keys", None):
        logger.warning("Unexpected keys in checkpoint: %s", res.unexpected_keys)

    if args.cache_dir:
        try:
            recreate_missing_cache(
                args.cache_dir, ds.items, s3tok, s3gen, batch_size=32
            )
        except Exception:
            logger.exception("Failed while recreating missing cache entries")

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_t3(b, s3tok, en_tok),
    )

    t3 = T3()
    load_t3_from_ckpt(t3, args.ckpt_dir)

    try:
        tok_vocab = len(en_tok.tokenizer.get_vocab())
    except Exception:
        tok_vocab = getattr(en_tok.tokenizer, "get_vocab_size", lambda: None)() or None

    try:
        if tok_vocab is not None and hasattr(t3, "text_emb"):
            cur_size = getattr(t3.text_emb, "num_embeddings", None)
            if cur_size is None and hasattr(t3.text_emb, "weight"):
                cur_size = t3.text_emb.weight.size(0)
            if cur_size is not None and tok_vocab > cur_size:
                old_emb = t3.text_emb
                new_emb = torch.nn.Embedding(tok_vocab, old_emb.embedding_dim)
                with torch.no_grad():
                    new_emb.weight[:cur_size].data.copy_(old_emb.weight[:cur_size].data)
                    try:
                        std = float(old_emb.weight.std().item())
                    except Exception:
                        std = 0.02
                    torch.nn.init.normal_(new_emb.weight[cur_size:], mean=0.0, std=std)
                t3.text_emb = new_emb
                if hasattr(t3, "text_head"):
                    old_head = t3.text_head
                    hidden = old_head.in_features
                    new_head = torch.nn.Linear(
                        hidden, tok_vocab, bias=getattr(old_head, "bias") is not None
                    )
                    with torch.no_grad():
                        ncopy = min(
                            getattr(old_head, "out_features", old_head.weight.size(0)),
                            tok_vocab,
                        )
                        new_head.weight[:ncopy].data.copy_(old_head.weight[:ncopy].data)
                        if tok_vocab > ncopy:
                            try:
                                std = float(old_head.weight.std().item())
                            except Exception:
                                std = 0.02
                            torch.nn.init.normal_(
                                new_head.weight[ncopy:], mean=0.0, std=std
                            )
                        if (
                            getattr(old_head, "bias", None) is not None
                            and new_head.bias is not None
                        ):
                            new_head.bias[:ncopy].data.copy_(old_head.bias[:ncopy].data)
                    t3.text_head = new_head
    except Exception:
        logger.exception("Failed while ensuring embeddings match tokenizer")
        raise

    # Auto-detect LoRA target module name substrings from the T3 model
    candidate_tokens = set()
    for name, mod in t3.named_modules():
        parts = name.split(".") if name else []
        for p in parts[-2:]:
            # avoid adding a generic "mlp" token which may refer to a composite module
            # (e.g. LlamaMLP) that contains Linear children; targeting the inner
            # proj/linear names is safer and supported by PEFT.
            if p == "mlp":
                continue
            if any(
                x in p
                for x in (
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "proj",
                    "linear",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                )
            ):
                candidate_tokens.add(p)
        cls = mod.__class__.__name__.lower()
        if "linear" in cls or "dense" in cls:
            candidate_tokens.add("linear")
        if "proj" in cls:
            candidate_tokens.add("proj")
    if not candidate_tokens:
        candidate_tokens = {"q_proj", "v_proj"}
    target_modules = list(candidate_tokens)
    logger.info("Auto-detected target_modules for T3 LoRA: %s", target_modules)

    lora_config = LoraConfig(
        r=8, lora_alpha=32, target_modules=target_modules, inference_mode=False
    )

    if args.single_device_transformer and device.type == "cuda":
        try:
            t3.tfmr.to(device)
        except Exception:
            try:
                t3.tfmr = t3.tfmr.to(device)
            except Exception:
                logger.exception("Failed to move transformer to device")
        try:
            if hasattr(t3, "text_emb"):
                t3.text_emb.to(device)
            if hasattr(t3, "text_head"):
                t3.text_head.to(device)
        except Exception:
            logger.exception("Failed to move embeddings/heads to device")
        model = get_peft_model(t3, lora_config)
    else:
        try:
            t3.to("cpu")
        except Exception:
            pass
        model = get_peft_model(t3, lora_config)
        if device.type == "cuda":
            gpu_dev = str(device)
            try:
                if args.auto_offload:
                    try:
                        moved_layers = t3.auto_offload(gpu_device=gpu_dev)
                        if moved_layers > 0:
                            logger.info(
                                "Auto-offloaded %d transformer layers to %s",
                                moved_layers,
                                gpu_dev,
                            )
                    except Exception:
                        logger.exception("auto_offload failed")
                layers_ref = None
                try:
                    layers_ref = getattr(t3.tfmr, "model", t3.tfmr).layers
                except Exception:
                    layers_ref = None
                if layers_ref is not None:
                    for i, layer in enumerate(list(layers_ref)):
                        try:
                            has_lora = any(
                                "lora" in n.lower()
                                for n, _ in layer.named_parameters(recurse=True)
                            )
                        except Exception:
                            has_lora = False
                        if has_lora:
                            try:
                                t3.set_offload(i, i + 1, gpu_device=gpu_dev)
                            except Exception:
                                pass
            except Exception:
                logger.exception("Failed to apply selective layer offload for LoRA")

    made_trainable = []
    for name, p in model.named_parameters():
        if "lora" in name.lower() and not p.requires_grad:
            p.requires_grad_(True)
            made_trainable.append(name)
    if made_trainable:
        logger.info("Set requires_grad=True on %d LoRA params", len(made_trainable))

    base_wrapper = getattr(model, "base_model", model)
    for layer_name in ("text_emb", "text_head"):
        layer = getattr(base_wrapper, layer_name, None)
        if layer is not None:
            for pname, p in layer.named_parameters():
                if not p.requires_grad:
                    p.requires_grad_(True)
                    made_trainable.append(f"{layer_name}.{pname}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters available for training")

    optim = torch.optim.AdamW(trainable_params, lr=args.lr)
    tb_writer = SummaryWriter(log_dir=str(Path(args.out_dir) / "tb"))
    global_step = 0
    model.train()
    try:
        model.to(device)
    except Exception:
        pass

    grad_accum = max(1, int(args.grad_accum_steps))
    use_fp16 = getattr(args, "fp16", False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    accum_counter = 0
    for epoch in range(args.epochs):
        loop = tqdm(dl, desc=f"epoch {epoch}")
        for step, batch in enumerate(loop):
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            txt = batch.get("text_tokens")
            if txt is None:
                raise RuntimeError("Batch missing text_tokens")
            if not torch.is_tensor(txt):
                txt = torch.as_tensor(txt, dtype=torch.long).to(device)
            else:
                txt = txt.long().to(device)

            base = getattr(model, "base_model", model)
            emb_layer = getattr(base, "text_emb", None)
            if emb_layer is not None:
                emb_size = getattr(emb_layer, "num_embeddings", None)
                if emb_size is None and hasattr(emb_layer, "weight"):
                    emb_size = emb_layer.weight.size(0)
                max_id = int(txt.max()) if txt.numel() else -1
                if emb_size is not None and max_id >= emb_size:
                    new_size = max(max_id + 1, emb_size)
                    old_emb = emb_layer
                    new_emb = torch.nn.Embedding(new_size, old_emb.embedding_dim)
                    with torch.no_grad():
                        ncopy = min(emb_size, new_size)
                        new_emb.weight[:ncopy].data.copy_(old_emb.weight[:ncopy].data)
                        if new_size > emb_size:
                            try:
                                std = float(old_emb.weight.std().item())
                            except Exception:
                                std = 0.02
                            torch.nn.init.normal_(
                                new_emb.weight[ncopy:], mean=0.0, std=std
                            )
                    if hasattr(model, "base_model"):
                        model.base_model.text_emb = new_emb
                    else:
                        setattr(model, "text_emb", new_emb)

            B = txt.size(0)
            txt_len_padded = int(txt.size(1))
            txt_lens = torch.full((B,), txt_len_padded, dtype=torch.long, device=device)

            speech_tokens = batch.get("speech_tokens")
            if speech_tokens is None:
                raise RuntimeError("Batch missing speech_tokens")
            if not torch.is_tensor(speech_tokens):
                speech_tokens = torch.as_tensor(
                    speech_tokens, dtype=torch.long, device=device
                )
            else:
                speech_tokens = speech_tokens.long().to(device)
            speech_len_padded = int(speech_tokens.size(1))
            speech_lens = torch.full(
                (B,), speech_len_padded, dtype=torch.long, device=speech_tokens.device
            )

            try:
                from chatterbox.models.t3.modules.cond_enc import T3Cond

                spk_dim = (
                    getattr(t3, "hp", None)
                    and getattr(t3.hp, "speaker_embed_size", None)
                ) or 256
                speaker_emb = torch.zeros(B, spk_dim, device=device)
                emotion_tensor = torch.full((B, 1, 1), float(0.5), device=device)
                t3_cond = T3Cond(speaker_emb=speaker_emb, emotion_adv=emotion_tensor)
            except Exception:

                class _FallbackCond:
                    def __init__(self, speaker_emb, emotion_adv):
                        self.speaker_emb = speaker_emb
                        self.clap_emb = None
                        self.cond_prompt_speech_tokens = None
                        self.cond_prompt_speech_emb = None
                        self.emotion_adv = emotion_adv

                spk_dim = (
                    getattr(t3, "hp", None)
                    and getattr(t3.hp, "speaker_embed_size", None)
                ) or 256
                speaker_emb = torch.zeros(B, spk_dim, device=device)
                emotion_tensor = torch.full((B, 1, 1), float(0.5), device=device)
                t3_cond = _FallbackCond(speaker_emb, emotion_tensor)

            def _call_loss(t3_cond, txt, txt_lens, speech_tokens, speech_token_lens):
                if hasattr(model, "loss"):
                    return model.loss(
                        t3_cond=t3_cond,
                        text_tokens=txt,
                        text_token_lens=txt_lens,
                        speech_tokens=speech_tokens,
                        speech_token_lens=speech_token_lens,
                    )
                else:
                    return t3.loss(
                        t3_cond=t3_cond,
                        text_tokens=txt,
                        text_token_lens=txt_lens,
                        speech_tokens=speech_tokens,
                        speech_token_lens=speech_token_lens,
                    )

            try:
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    try:
                        loss_res = _call_loss(
                            t3_cond, txt, txt_lens, speech_tokens, speech_lens
                        )
                    except AssertionError:
                        B = speech_tokens.size(0)
                        speech_lens = torch.full(
                            (B,),
                            int(speech_tokens.size(1)),
                            dtype=torch.long,
                            device=speech_tokens.device,
                        )
                        loss_res = _call_loss(
                            t3_cond, txt, txt_lens, speech_tokens, speech_lens
                        )
                    # handle separate text and speech losses when returned as tuple/list
                    if isinstance(loss_res, (tuple, list)):
                        try:
                            lt_raw, ls_raw = loss_res
                        except Exception:
                            # fallback: sum all and split evenly
                            total = sum(
                                (
                                    l
                                    if torch.is_tensor(l)
                                    else torch.tensor(float(l), device=device)
                                )
                                for l in loss_res
                            )
                            lt = total * 0.5
                            ls = total * 0.5
                        else:
                            lt = (
                                lt_raw
                                if torch.is_tensor(lt_raw)
                                else torch.tensor(float(lt_raw), device=device)
                            )
                            ls = (
                                ls_raw
                                if torch.is_tensor(ls_raw)
                                else torch.tensor(float(ls_raw), device=device)
                            )
                    else:
                        # single loss returned: split evenly between text and speech
                        total = (
                            loss_res
                            if torch.is_tensor(loss_res)
                            else torch.tensor(float(loss_res), device=device)
                        )
                        lt = total * 0.5
                        ls = total * 0.5
                    # combined loss for backward
                    loss = lt + ls
                    if not torch.is_tensor(loss):
                        loss = torch.tensor(float(loss), device=device)
                    # scale for gradient accumulation
                    loss = loss / float(grad_accum)
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    accum_counter += 1
                    if accum_counter % grad_accum == 0:
                        if scaler is not None:
                            scaler.unscale_(optim)
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                        if scaler is not None:
                            scaler.step(optim)
                            scaler.update()
                        else:
                            optim.step()
                        optim.zero_grad()
                        # report un-divided losses (per-step)
                        report_loss = (loss * float(grad_accum)).detach()
                        report_loss_text = (lt * float(grad_accum)).detach()
                        report_loss_speech = (ls * float(grad_accum)).detach()
                        loop.set_postfix(
                            loss=float(report_loss.cpu().item()),
                            loss_text=float(report_loss_text.cpu().item()),
                            loss_speech=float(report_loss_speech.cpu().item()),
                        )
                        try:
                            tb_writer.add_scalar(
                                "train/loss",
                                float(report_loss.cpu().item()),
                                global_step,
                            )
                            tb_writer.add_scalar(
                                "train/loss_text",
                                float(report_loss_text.cpu().item()),
                                global_step,
                            )
                            tb_writer.add_scalar(
                                "train/loss_speech",
                                float(report_loss_speech.cpu().item()),
                                global_step,
                            )
                        except Exception:
                            logger.exception("Failed to write loss to TensorBoard")
                        global_step += 1
                        try:
                            if global_step % 2000 == 0:
                                Path(args.out_dir).mkdir(parents=True, exist_ok=True)
                                peft_state = get_peft_model_state_dict(model)
                                torch.save(
                                    peft_state,
                                    Path(args.out_dir)
                                    / f"lora_state_step_{global_step}.pt",
                                )
                                base_updates = {}
                                try:
                                    base_updates["text_emb"] = t3.text_emb.state_dict()
                                except Exception:
                                    pass
                                try:
                                    base_updates["text_head"] = (
                                        t3.text_head.state_dict()
                                    )
                                except Exception:
                                    pass
                                if base_updates:
                                    torch.save(
                                        base_updates,
                                        Path(args.out_dir)
                                        / f"base_updates_step_{global_step}.pt",
                                    )
                                logger.info(
                                    "Saved periodic checkpoint at step %d to %s",
                                    global_step,
                                    args.out_dir,
                                )
                        except Exception:
                            logger.exception("Failed to save periodic checkpoint")
            except Exception:
                logger.exception("Failed during training step")
                raise

    if accum_counter % grad_accum != 0:
        try:
            if scaler is not None:
                scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            if scaler is not None:
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()
            optim.zero_grad()
        except Exception:
            logger.exception("Failed during final accumulation step")
            raise

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    peft_state = get_peft_model_state_dict(model)
    torch.save(peft_state, Path(args.out_dir) / "lora_t3_state.pt")
    base_updates = {}
    try:
        base_updates["text_emb"] = t3.text_emb.state_dict()
    except Exception:
        pass
    try:
        base_updates["text_head"] = t3.text_head.state_dict()
    except Exception:
        pass
    if base_updates:
        torch.save(base_updates, Path(args.out_dir) / "base_updates.pt")


if __name__ == "__main__":
    main()
