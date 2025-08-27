import torch
import torchaudio as ta

from chatterbox.tts import ChatterboxTTS

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cpu"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

model = ChatterboxTTS.from_pretrained(device=device)

text = "Merhaba! Bugün nasılsın? Bu cümle Türkçe ses sentezi için bir testtir."
wav = model.generate(text)
ta.save("test-1.wav", wav, model.sr)
