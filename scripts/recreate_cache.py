from dataset_utils import recreate_missing_cache, read_meta

from chatterbox import ChatterboxTTS
import argparse

p = argparse.ArgumentParser()
p.add_argument("--ckpt_dir", default="ckpt")
p.add_argument("--cache_dir", default="cache")
p.add_argument("--meta", default="metadata.tsv")
p.add_argument("--batch_size", default=32)
p.add_argument("--device", default="cpu")
args = p.parse_args()


tts = ChatterboxTTS.from_local(args.ckpt_dir, args.device)

meta = read_meta(args.meta)

recreate_missing_cache(
    args.cache_dir,
    meta,
    tts.s3gen.tokenizer,
    tts.ve,
    tts.s3gen,
    args.batch_size,
    args.device,
)
