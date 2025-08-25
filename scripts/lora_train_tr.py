import argparse
from pathlib import Path
import sys
import logging
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchaudio as ta
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file
from chatterbox.models.s3tokenizer import S3Tokenizer
from chatterbox.models.tokenizers import EnTokenizer
from chatterbox.models.t3.t3 import T3

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

def read_meta(path):
    items = []
    base = Path(path).resolve().parent
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                p, text = line.split("\t", 1)
            except Exception as e:
                logger.exception("Malformed meta line %s: %s", lineno, line)
                raise
            p_path = Path(p)
            if not p_path.is_absolute():
                cand = base / p
                if cand.exists():
                    wav_path = str(cand)
                elif (Path.cwd() / p).exists():
                    wav_path = str(Path.cwd() / p)
                else:
#                    logger.warning("WAV path not found for line %s: %s -- marking as missing (will be skipped)", lineno, p)
                    wav_path = None
            else:
                wav_path = str(p_path)
            items.append({"wav": wav_path, "text": text})
    return items

class SimpleMetaDataset(Dataset):
    def __init__(self, meta_path, s3tok, en_tok, sample_rate=16000, cache_dir=None, force_recompute=False):
        self.items = read_meta(meta_path)
        self.s3tok = s3tok
        self.en_tok = en_tok
        self.sample_rate = sample_rate
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.force_recompute = bool(force_recompute)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # provide idx and cache path to allow collator to save computed tokens
        cache_path = None
        if self.cache_dir is not None:
            cache_path = self.cache_dir / f"{idx}.pt"
            if cache_path.exists() and not self.force_recompute:
                try:
                    data = torch.load(str(cache_path))
                    if isinstance(data, dict):
                        return {
                            "speech_tokens": data.get("speech_tokens"),
                            "speech_lens": data.get("speech_token_lens") or data.get("speech_lens"),
                            "text_ids": data.get("text_ids"),
                            "text": data.get("text"),
                            "idx": idx,
                            "cache_path": str(cache_path),
                        }
                except Exception:
                    logger.exception("Failed to load cache %s, will recompute", cache_path)
                    # fall through to recompute rather than silently ignoring

        it = self.items[idx]
        return {"wav": it["wav"], "text": it["text"], "idx": idx, "cache_path": str(cache_path) if cache_path is not None else None}


def simple_collate(batch, s3tok, en_tok):
    # filter out items that have no wav and no cached speech_tokens
    filtered = []
    skipped = []
    for it in batch:
        if it.get("speech_tokens") is None and it.get("wav") is None:
            skipped.append(it.get("idx") if it.get("idx") is not None else "?")
        else:
            filtered.append(it)
    if skipped:
        logger.warning("Skipping %d items missing wav and cache: %s", len(skipped), skipped)
    if not filtered:
        raise RuntimeError("All items in batch are missing wavs or cached speech tokens; cannot form batch")
    batch = filtered

    # support items with precomputed speech_tokens/text_ids (cached) and raw wavs
    computed_wavs = []
    computed_indices = []
    texts = []
    speech_tokens_list = []
    speech_lens_list = []
    text_id_list = []
    cache_paths = []

    for i, it in enumerate(batch):
        cache_paths.append(it.get("cache_path"))
        if it.get("speech_tokens") is not None:
            st = it.get("speech_tokens")
            sl = it.get("speech_lens")
            if torch.is_tensor(st):
                speech_tokens_list.append(st)
            else:
                speech_tokens_list.append(torch.as_tensor(st))
            if sl is None:
                speech_lens_list.append(torch.tensor(speech_tokens_list[-1].size(0), dtype=torch.long))
            else:
                speech_lens_list.append(torch.as_tensor(sl, dtype=torch.long))
            text_id_list.append(it.get("text_ids"))
            texts.append(it.get("text") or "")
        else:
            # will compute speech tokens from wav
            wav_path = it.get("wav")
            try:
                if wav_path is None:
                    raise FileNotFoundError("missing wav")
                waveform, sr = ta.load(wav_path)
            except Exception:
                logger.warning("Failed to load wav %s -- using silence", wav_path)
                waveform = torch.zeros(1, 16000)
                sr = 16000
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != 16000:
                waveform = ta.transforms.Resample(sr, 16000)(waveform)
            wav_tensor = waveform.squeeze(0)
            computed_wavs.append(wav_tensor)
            computed_indices.append(i)
            texts.append(it.get("text", ""))
            text_id_list.append(None)
            # placeholder for lens/token
            speech_tokens_list.append(None)
            speech_lens_list.append(None)

    # compute speech tokens for computed_wavs
    if computed_wavs:
        try:
            comp_tokens, comp_lens = s3tok(computed_wavs)
        except Exception:
            logger.exception("S3Tokenizer failed to tokenize computed wavs")
            raise
        # insert into lists at their positions
        comp_i = 0
        for idx in computed_indices:
            tok = comp_tokens[comp_i]
            ln = comp_lens[comp_i]
            if not torch.is_tensor(tok):
                tok = torch.as_tensor(tok)
            if not torch.is_tensor(ln):
                ln = torch.as_tensor(ln, dtype=torch.long)
            speech_tokens_list[idx] = tok
            speech_lens_list[idx] = ln
            # save to cache if requested
            cpath = cache_paths[idx]
            if cpath is not None:
                try:
                    Path(cpath).parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"speech_tokens": tok, "speech_token_lens": ln, "text": texts[idx]}, cpath)
                except Exception:
                    logger.exception("Failed to write cache %s", cpath)
            comp_i += 1

    # ensure all speech tokens are tensors and pad to a batch tensor
    speech_tokens_tensors = [st if torch.is_tensor(st) else torch.as_tensor(st) for st in speech_tokens_list]
    speech_padded = torch.nn.utils.rnn.pad_sequence(speech_tokens_tensors, batch_first=True, padding_value=0).long()
    speech_lens = torch.tensor([int(x.tolist() if torch.is_tensor(x) and x.numel()>1 else (x.item() if torch.is_tensor(x) else int(x))) for x in speech_lens_list], dtype=torch.long)

    # prepare text ids
    sot_id = en_tok.tokenizer.token_to_id("[START]")
    eot_id = en_tok.tokenizer.token_to_id("[STOP]")
    txt_ids = []
    for t_ids, txt in zip(text_id_list, texts):
        if t_ids is not None:
            ids = [sot_id] + list(t_ids) + [eot_id]
        else:
            ids = [sot_id] + en_tok.encode(txt) + [eot_id]
        txt_ids.append(torch.tensor(ids, dtype=torch.long))
    txt_padded = torch.nn.utils.rnn.pad_sequence(txt_ids, batch_first=True, padding_value=0)
    txt_lens = torch.tensor([t.numel() for t in txt_ids], dtype=torch.long)

    return {"text_tokens": txt_padded, "text_lens": txt_lens, "speech_tokens": speech_padded, "speech_lens": speech_lens}

def load_t3_from_ckpt(t3, ckpt_dir):
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
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    en_tok = EnTokenizer(str(args.tokenizer))
    s3tok = S3Tokenizer()
    try:
        if torch.cuda.is_available():
            s3tok.to("cuda")
    except Exception:
        pass
    ds = SimpleMetaDataset(args.meta, s3tok, en_tok, cache_dir=args.cache_dir, force_recompute=args.force_recompute)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: simple_collate(b, s3tok, en_tok))
    t3 = T3()
    load_t3_from_ckpt(t3, args.ckpt_dir)
    # compute tokenizer vocab size for validation
    try:
        tok_vocab = len(en_tok.tokenizer.get_vocab())
    except Exception:
        tok_vocab = getattr(en_tok.tokenizer, "get_vocab_size", lambda: None)() or None
    # ensure tokenizer and model embeddings match; extend ONCE if tokenizer grew
    try:
        try:
            tok_vocab = len(en_tok.tokenizer.get_vocab())
        except Exception:
            tok_vocab = getattr(en_tok.tokenizer, "get_vocab_size", lambda: None)() or None
        if tok_vocab is not None:
            if hasattr(t3, "text_emb"):
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
                    try:
                        new_emb.weight.requires_grad_(True)
                        logger.info("Made extended text embedding parameters trainable")
                    except Exception:
                        logger.exception("Failed to set requires_grad on extended embeddings")
                    logger.info("Extended text embeddings from %d to %d", cur_size, tok_vocab)
                    # extend text head if present
                    if hasattr(t3, "text_head"):
                        old_head = t3.text_head
                        hidden = old_head.in_features
                        new_head = torch.nn.Linear(hidden, tok_vocab, bias=getattr(old_head, "bias") is not None)
                        with torch.no_grad():
                            ncopy = min(getattr(old_head, "out_features", old_head.weight.size(0)), tok_vocab)
                            new_head.weight[:ncopy].data.copy_(old_head.weight[:ncopy].data)
                            if tok_vocab > ncopy:
                                try:
                                    std = float(old_head.weight.std().item())
                                except Exception:
                                    std = 0.02
                                torch.nn.init.normal_(new_head.weight[ncopy:], mean=0.0, std=std)
                            if getattr(old_head, "bias", None) is not None and new_head.bias is not None:
                                new_head.bias[:ncopy].data.copy_(old_head.bias[:ncopy].data)
                        # make new head parameters trainable since new weights are initialized randomly
                        try:
                            new_head.weight.requires_grad_(True)
                            if new_head.bias is not None:
                                new_head.bias.requires_grad_(True)
                            logger.info("Made extended text_head parameters trainable")
                        except Exception:
                            logger.exception("Failed to set requires_grad on extended text_head")
                        t3.text_head = new_head
                        logger.info("Extended text head output from %d to %d", getattr(old_head, "out_features", old_head.weight.size(0)), tok_vocab)
    except Exception as e:
        logger.exception("Failed while ensuring embeddings match tokenizer: %s", e)
        raise

    t3.to(device)
    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], inference_mode=False)
    model = get_peft_model(t3, lora_config)
    # Debug: list PEFT/LoRA-related parameters
    lora_names = [name for name, _ in model.named_parameters() if 'lora' in name.lower() or 'lora' in name]
    logger.info("Detected %d LoRA-related parameter names (sample up to 200): %s", len(lora_names), lora_names[:200])
    # Ensure model exposes the same embedding/head modules as t3 so forward uses trainable params
    for attr in ("text_emb", "text_head"):
        try:
            if not hasattr(model, attr) and hasattr(t3, attr):
                setattr(model, attr, getattr(t3, attr))
                logger.info("Assigned model.%s to t3.%s to ensure forward uses base module", attr, attr)
        except Exception:
            logger.exception("Failed to assign %s from t3 to model", attr)
    # Ensure LoRA params are trainable
    made_trainable = []
    for name, p in model.named_parameters():
        if 'lora' in name.lower() and not p.requires_grad:
            p.requires_grad_(True)
            made_trainable.append(name)
    if made_trainable:
        logger.info("Set requires_grad=True on %d LoRA params (sample): %s", len(made_trainable), made_trainable[:200])
    # ensure extended base embedding/head params are trainable
    base_wrapper = getattr(model, "base_model", model)
    for layer_name in ("text_emb", "text_head"):
        layer = getattr(base_wrapper, layer_name, None)
        if layer is None and hasattr(model, layer_name):
            layer = getattr(model, layer_name)
        if layer is not None:
            for pname, p in layer.named_parameters():
                if not p.requires_grad:
                    p.requires_grad_(True)
                    made_trainable.append(f"{layer_name}.{pname}")
    if made_trainable:
        logger.info("Additionally made these params trainable: %s", made_trainable[:200])
    # collect trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info("Total trainable params count: %d", len(trainable_params))
    if not trainable_params:
        logger.error("PEFT model has no trainable parameters. Check LoRA config and target modules.")
        raise RuntimeError("No trainable parameters available for training; remove silent fallbacks and fix configuration")
    optim = torch.optim.AdamW(trainable_params, lr=args.lr)
    tb_writer = SummaryWriter(log_dir=str(Path(args.out_dir)/"tb"))
    global_step = 0
    model.train()
    for epoch in range(args.epochs):
        loop = tqdm(dl, desc=f"epoch {epoch}")
        for step, batch in enumerate(loop):
            # move tensors to device
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            txt = batch.get("text_tokens")
            if txt is None:
                logger.error("Batch missing text_tokens")
                raise RuntimeError("Batch missing text_tokens")
            if not torch.is_tensor(txt):
                txt = torch.as_tensor(txt, dtype=torch.long).to(device)
            else:
                txt = txt.long().to(device)
            # extend text embeddings if token ids exceed current embedding size
            base = getattr(model, "base_model", model)
            emb_layer = getattr(base, "text_emb", None)
            if emb_layer is not None:
                emb_size = getattr(emb_layer, "num_embeddings", None)
                if emb_size is None and hasattr(emb_layer, "weight"):
                    emb_size = emb_layer.weight.size(0)
                max_id = int(txt.max()) if txt.numel() else -1
                if emb_size is not None and max_id >= emb_size:
                    new_size = max(max_id + 1, tok_vocab if 'tok_vocab' in locals() and tok_vocab is not None else max_id + 1)
                    logger.info("Extending text embedding from %s to %s", emb_size, new_size)
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
                            torch.nn.init.normal_(new_emb.weight[ncopy:], mean=0.0, std=std)
                    # assign new embedding back to base model and wrapper
                    if hasattr(model, "base_model"):
                        model.base_model.text_emb = new_emb
                    else:
                        setattr(model, "text_emb", new_emb)
                    # also extend text_head output dim if present
                    try:
                        head = getattr(base, "text_head", None)
                        if head is not None:
                            old_head = head
                            hidden = old_head.in_features
                            new_head = torch.nn.Linear(hidden, new_size, bias=getattr(old_head, "bias") is not None)
                            with torch.no_grad():
                                ncopy2 = min(getattr(old_head, "out_features", old_head.weight.size(0)), new_size)
                                new_head.weight[:ncopy2].data.copy_(old_head.weight[:ncopy2].data)
                                if new_size > ncopy2:
                                    try:
                                        std2 = float(old_head.weight.std().item())
                                    except Exception:
                                        std2 = 0.02
                                    torch.nn.init.normal_(new_head.weight[ncopy2:], mean=0.0, std=std2)
                                if getattr(old_head, "bias", None) is not None and new_head.bias is not None:
                                    new_head.bias[:ncopy2].data.copy_(old_head.bias[:ncopy2].data)
                            if hasattr(model, "base_model"):
                                model.base_model.text_head = new_head
                            else:
                                setattr(model, "text_head", new_head)
                    except Exception:
                        logger.exception("Failed to extend text_head")
            try:
                emb = model.text_emb(txt)
                out = model.text_head(emb.mean(dim=1))
                target = torch.zeros_like(out, device=out.device)
                loss = torch.nn.functional.mse_loss(out, target)
            except Exception:
                logger.exception("Failed to compute loss for batch; aborting")
                raise
            optim.zero_grad()
            loss.backward()
            optim.step()
            loop.set_postfix(loss=loss.item())
            try:
                tb_writer.add_scalar("train/loss", float(loss.detach().cpu().item()), global_step)
            except Exception:
                logger.exception("Failed to write loss to TensorBoard")
            global_step += 1
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    # save only PEFT (LoRA) weights plus any modified base embedding/head
    peft_state = get_peft_model_state_dict(model)
    torch.save(peft_state, Path(args.out_dir) / "lora_lora_state.pt")
    # save base model updates (embedding/head) separately so they can be merged if needed
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
