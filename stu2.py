from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import os

# download and load all models
preload_models()

# # generate audio from text
# text_prompt = """
#      ♪  one two three four five six lalala  ♪
# """
# audio_array = generate_audio(text_prompt)
# # save audio to disk
# write_wav("bark_generation.wav", SAMPLE_RATE, audio_array)

for text_prompt in [
    "♪ In the jungle, the mighty jungle, the lion barks tonight ♪",
    "♪ one two three four five six lalala ♪",
    "♪ where can I go to find old days ♪",
]:
    for tt in range(3, 9):
        for wt in range(3, 9):
            if os.path.isfile(f"{text_prompt}_{tt}_{wt}.wav"):
                continue
            audio_array = generate_audio(
                text_prompt,
                text_temp=0.1*tt,
                waveform_temp=0.1*wt,
            )

            # save audio to disk
            write_wav(f"{text_prompt}_{tt}_{wt}.wav", SAMPLE_RATE, audio_array)
