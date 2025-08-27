import logging
from pathlib import Path

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
            items.append({"wav": wav_path, "text": text})
    return items


class SimpleMetaDataset(Dataset):
    def __init__(
        self, meta_path, s3tok, sample_rate=16000, cache_dir=None, force_recompute=False
    ):
        self.items = read_meta(meta_path)
        self.s3tok = s3tok
        self.sample_rate = sample_rate
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.force_recompute = bool(force_recompute)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        cache_path = None
        if self.cache_dir is not None:
            cache_path = self.cache_dir / f"{idx}.pt"
            if cache_path.exists() and not self.force_recompute:
                try:
                    data = torch.load(str(cache_path), map_location="cpu")
                    if isinstance(data, dict):
                        return {
                            "speech_tokens": data.get("speech_tokens"),
                            "speech_lens": data.get("speech_token_lens")
                            or data.get("speech_lens"),
                            "text": data.get("text"),
                            "idx": idx,
                            "cache_path": str(cache_path),
                            "wav": data.get("wav") or self.items[idx].get("wav"),
                        }
                except Exception:
                    logger.exception(
                        "Failed to load cache %s, will recompute", cache_path
                    )
        it = self.items[idx]
        return {
            "wav": it.get("wav"),
            "text": it.get("text"),
            "idx": idx,
            "cache_path": str(cache_path) if cache_path is not None else None,
        }


def collate_s3gen(batch, s3tok, s3gen):
    tokens = []
    token_lens = []
    feats = []
    feat_lens = []
    embeddings = []

    for it in batch:
        wav_path = it.get("wav")
        speech_tokens = it.get("speech_tokens")
        speech_lens = it.get("speech_lens")
        # load wav
        if wav_path is None:
            waveform = torch.zeros(1, 16000)
            sr = 16000
        else:
            try:
                waveform, sr = ta.load(wav_path)
            except Exception:
                logger.exception("Failed to load wav %s -- using silence", wav_path)
                waveform = torch.zeros(1, 16000)
                sr = 16000
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = ta.transforms.Resample(sr, 16000)(waveform)
        wav16 = waveform.squeeze(0)

        # speech tokens: prefer cache
        if speech_tokens is None:
            try:
                st, sl = s3tok([wav16])
                st = st[0]
                sl = sl[0]
            except Exception:
                logger.exception("S3Tokenizer failed for wav -- using empty tokens")
                st = torch.zeros(1, dtype=torch.long)
                sl = torch.tensor(1, dtype=torch.long)
        else:
            st = speech_tokens
            sl = speech_lens
            if not torch.is_tensor(st):
                st = torch.as_tensor(st, dtype=torch.long)
            if not torch.is_tensor(sl):
                sl = torch.as_tensor(sl, dtype=torch.long)
        tokens.append(st)
        token_lens.append(
            int(
                sl.tolist()
                if torch.is_tensor(sl) and sl.numel() > 1
                else (sl.item() if torch.is_tensor(sl) else int(sl))
            )
        )

        # compute mel at 24k and embedding
        try:
            wav24 = ta.transforms.Resample(16000, 24000)(wav16.unsqueeze(0)).squeeze(0)
        except Exception:
            wav24 = wav16
        mel = s3gen.mel_extractor(wav24.unsqueeze(0)).transpose(1, 2)  # (1, T, n_mels)
        feats.append(mel.squeeze(0))
        feat_lens.append(mel.shape[1] if mel.dim() == 3 else mel.size(1))

        # embedding via speaker encoder using 16k
        try:
            emb = s3gen.speaker_encoder.inference(wav16.unsqueeze(0))
        except Exception:
            emb = torch.zeros(
                1,
                (
                    s3gen.speaker_encoder.output_size()
                    if hasattr(s3gen.speaker_encoder, "output_size")
                    else 192
                ),
            )
        embeddings.append(emb.squeeze(0))

    padded_tokens = torch.nn.utils.rnn.pad_sequence(
        tokens, batch_first=True, padding_value=0
    ).long()
    token_lens_tensor = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)

    max_T = max(f.size(0) for f in feats)
    n_mels = feats[0].size(1)
    padded_feats = torch.zeros(len(feats), max_T, n_mels)
    for i, f in enumerate(feats):
        padded_feats[i, : f.size(0), :] = f
    feat_lens_tensor = torch.tensor([int(x) for x in feat_lens], dtype=torch.long)

    embeddings_tensor = torch.stack(embeddings, dim=0)

    out = {
        "speech_token": padded_tokens,
        "speech_token_len": token_lens_tensor,
        "speech_feat": padded_feats,
        "speech_feat_len": feat_lens_tensor,
        "embedding": embeddings_tensor,
    }
    return out


def recreate_missing_cache(
    cache_dir, meta_items, s3tok, batch_size=32, sample_rate=16000
):
    cache_dir = Path(cache_dir)
    total = len(meta_items)
    missing = [idx for idx in range(total) if not (cache_dir / f"{idx}.pt").exists()]
    logger.info(
        "Using cache_dir=%s : %d/%d cached, %d missing",
        str(cache_dir),
        total - len(missing),
        total,
        len(missing),
    )
    if not missing:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    from tqdm import tqdm

    pbar = tqdm(total=len(missing), desc="Recreating cache", unit="file")
    for i in range(0, len(missing), batch_size):
        batch_idxs = missing[i : i + batch_size]
        wavs = []
        texts = []
        valid_idxs = []
        for idx in batch_idxs:
            item = meta_items[idx]
            wav_path = item.get("wav")
            text = item.get("text", "")
            if not wav_path:
                pbar.update(1)
                continue
            try:
                waveform, sr = ta.load(wav_path)
            except Exception:
                logger.exception(
                    "Failed to load wav %s for cache idx %d -- saving silence",
                    wav_path,
                    idx,
                )
                waveform = torch.zeros(1, sample_rate)
                sr = sample_rate
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != sample_rate:
                waveform = ta.transforms.Resample(sr, sample_rate)(waveform)
            wavs.append(waveform.squeeze(0))
            texts.append(text)
            valid_idxs.append(idx)
        if not valid_idxs:
            continue
        try:
            toks, lens = s3tok(wavs)
        except Exception:
            logger.exception(
                "S3Tokenizer failed while recreating cache for batch starting at %d", i
            )
            pbar.update(len(valid_idxs))
            continue
        for j, idx in enumerate(valid_idxs):
            tok = toks[j]
            ln = lens[j]
            if not torch.is_tensor(tok):
                tok = torch.as_tensor(tok)
            if not torch.is_tensor(ln):
                ln = torch.as_tensor(ln, dtype=torch.long)
            cpath = cache_dir / f"{idx}.pt"
            try:
                torch.save(
                    {
                        "speech_tokens": tok,
                        "speech_token_lens": ln,
                        "text": texts[j],
                        "wav": None,
                    },
                    str(cpath),
                )
            except Exception:
                logger.exception("Failed to write cache %s", str(cpath))
            pbar.update(1)
    pbar.close()
