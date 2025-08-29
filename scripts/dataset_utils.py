import logging
from pathlib import Path

from tqdm import tqdm
import torch
import torchaudio as ta
from torch.utils.data import Dataset

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
            except Exception:
                p = line
                text = ""
            p_path = Path(p)
            if not p_path.is_absolute():
                cand = base / p
                if cand.exists():
                    wav_path = str(cand)
                elif (Path.cwd() / p).exists():
                    wav_path = str(Path.cwd() / p)
                else:
                    wav_path = None
            else:
                wav_path = str(p_path)
            if wav_path is not None:
                items.append({"wav": wav_path, "text": text})
    return items


class SimpleMetaDataset(Dataset):
    def __init__(self, meta_path, cache_dir=None, force_recompute=False):
        self.items = read_meta(meta_path)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        cache_path = None
        if self.cache_dir is not None:
            cache_path = self.cache_dir / f"{idx}.pt"
            if cache_path.exists():
                try:
                    data = torch.load(str(cache_path))
                    if isinstance(data, dict):
                        return {
                            "speech_tokens": data["speech_tokens"],
                            "speech_lens": data["speech_token_lens"],
                            "text": data["text"],
                            "idx": idx,
                            "cache_path": str(cache_path),
                            "wav": data["wav"],
                            "embedding": data["embedding"],
                            "mel": data["mel"],
                            "mel_len": data["mel_len"],
                            "cache_used": True,
                        }
                except Exception:
                    logger.exception("Oh shit %s", cache_path)
        raise Exception("asgkjbasgfaerwgiugiuawbhgoiuhewiughaewiughiuewah")


def collate_s3gen(batch):
    toks = [b["speech_tokens"].long() for b in batch]
    toks_pad = torch.nn.utils.rnn.pad_sequence(toks, batch_first=True, padding_value=0)
    tok_lens = torch.tensor([t.size(0) for t in toks])

    emb = torch.stack([b["embedding"] for b in batch], dim=0)
    mels = torch.nn.utils.rnn.pad_sequence(
        [b["mel"][: b["mel_len"]] for b in batch], batch_first=True, padding_value=0
    )
    mel_lens = torch.tensor([b["mel_len"] for b in batch])

    out = {
        "speech_token": toks_pad,
        "speech_token_len": tok_lens,
        "speech_feat": mels,
        "speech_feat_len": mel_lens,
        "embedding": emb,
    }
    return out


def collate_t3(batch, en_tok):
    toks = [b["speech_tokens"].long() for b in batch]
    toks_pad = torch.nn.utils.rnn.pad_sequence(toks, batch_first=True, padding_value=0)
    tok_lens = torch.tensor([t.size(0) for t in toks])

    texts = [b["text"] for b in batch]

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
        "speech_tokens": toks_pad,
        "speech_lens": tok_lens,
    }
    return out


def recreate_missing_cache(
    cache_dir,
    meta_items,
    s3tok,
    s3gen,
    batch_size=32,
    device=None,
):
    """Recreate missing tokenizer caches on an optional device (e.g. 'cuda').

    If s3gen is provided, compute and store speaker embeddings for each cached item.
    """
    device = (
        torch.device(device)
        if device is not None
        else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    )
    cache_dir = Path(cache_dir)
    total = len(meta_items)
    missing = [idx for idx in range(total) if not (cache_dir / f"{idx}.pt").exists()]
    logger.info(
        "Using cache_dir=%s : %d/%d cached, %d missing (device=%s)",
        str(cache_dir),
        total - len(missing),
        total,
        len(missing),
        str(device),
    )
    if not missing:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)

    s3tok.to(device)
    s3gen.to(device)

    pbar = tqdm(total=len(missing), desc="Recreating cache", unit="file")
    for i in range(0, len(missing), batch_size):
        torch.cuda.empty_cache()
        batch_idxs = missing[i : i + batch_size]
        texts = []
        wavs = []
        wav_lens = []
        wavs_24k = []
        valid_idxs = []
        for idx in batch_idxs:
            item = meta_items[idx]
            wav_path = item.get("wav")
            text = item.get("text", "")
            if not wav_path:
                pbar.update(1)
                continue
            waveform, sr = ta.load(wav_path)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = ta.transforms.Resample(sr, 16000).to(device)(waveform.to(device))
            waveform_24k = ta.transforms.Resample(sr, 24000).to(device)(
                waveform.to(device)
            )

            wavs.append(waveform.squeeze(0).to(device))
            wav_lens.append(waveform_24k.size(1))
            wavs_24k.append(waveform_24k.squeeze(0).to(device))
            texts.append(text)
            valid_idxs.append(idx)
        if not valid_idxs:
            continue
        with torch.no_grad():
            toks, lens = s3tok(wavs)
            embs = s3gen.speaker_encoder.inference(wavs)

            wavs_24k = torch.nn.utils.rnn.pad_sequence(
                wavs_24k, batch_first=True, padding_value=0.0
            ).to(device)
            mels = s3gen.mel_extractor(wavs_24k).transpose(1, 2)

            feat_lens = torch.tensor(
                [int(mels.shape[1] * L / wavs_24k.size(1)) for L in wav_lens]
            )
        for j, idx in enumerate(valid_idxs):
            tok = toks[j].cpu()
            ln = lens[j].cpu()
            emb = embs[j].cpu()
            mel = mels[j].cpu()
            mel_len = feat_lens[j].cpu()

            cpath = cache_dir / f"{idx}.pt"
            try:
                save_obj = {
                    "speech_tokens": tok,
                    "speech_token_lens": ln,
                    "text": texts[j],
                    "wav": (
                        meta_items[idx].get("wav") if idx < len(meta_items) else None
                    ),
                    "embedding": emb,
                    "mel": mel,
                    "mel_len": mel_len,
                }
                torch.save(save_obj, str(cpath))
            except Exception:
                logger.exception("Failed to write cache %s", str(cpath))
            pbar.update(1)
    pbar.close()
