# https://colab.research.google.com/drive/1dWWkZzvu7L9Bunq9zvD-W02RFUXoW-Pd?usp=sharing#scrollTo=cWuW8SdgcO_Q
from transformers import AutoProcessor, BarkModel
import torch, os
import scipy

device = "cuda:0" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")
# model = BarkModel.from_pretrained("suno/bark-small")
model = model.to(device)

voice_preset = "v2/en_speaker_5"

inputs = processor(
    """
    ♪ where can I go to find old days ♪
""",
    voice_preset=voice_preset
)
# inputs = processor("[clears throat] Hello uh ..., my dog is cute [laughter]")
# # https://sunoaiwiki.com/en/resources/2024-05-13-list-of-metatags/

audio_array = model.generate(
    **inputs.to(device),
    # num_beams=4,
    # temperature=0.5,
    # semantic_temperature=0.8,
)
audio_array = audio_array.cpu().numpy().squeeze()

sample_rate = model.generation_config.sample_rate
scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=audio_array)

for te in range(3, 9):
    audio_array = model.generate(
        **inputs.to(device),
        temperature=0.1*te,
    )
    audio_array = audio_array.cpu().numpy().squeeze()

    sample_rate = model.generation_config.sample_rate
    scipy.io.wavfile.write(f"bark_out_temperature_{te}.wav", rate=sample_rate, data=audio_array)

for nb in range(3, 9):
    audio_array = model.generate(
        **inputs.to(device),
        num_beams=nb,
    )
    audio_array = audio_array.cpu().numpy().squeeze()

    sample_rate = model.generation_config.sample_rate
    scipy.io.wavfile.write(f"bark_out_num_beams_{nb}.wav", rate=sample_rate, data=audio_array)

for st in range(3, 9):
    audio_array = model.generate(
        **inputs.to(device),
        semantic_temperature=0.1*st,
    )
    audio_array = audio_array.cpu().numpy().squeeze()

    sample_rate = model.generation_config.sample_rate
    scipy.io.wavfile.write(f"bark_out_semantic_tem_{st}.wav", rate=sample_rate, data=audio_array)
