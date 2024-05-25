import torch, json, os, requests, time
from functools import wraps
from transformers import AutoProcessor, BarkModel
from bark import SAMPLE_RATE, generate_audio, preload_models
import nltk  # we'll use this to split into sentences
import numpy as np


streamlit: bool = False
temp: str = 'temp'  # file temp ban đầu khi chuyển từng đoạn thoại ra audio
historyfile: str = "hist.txt"
debug: bool = True

device = "cpu"  # "cuda:0" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")
# model = BarkModel.from_pretrained("suno/bark-small")
model = model.to(device)
sample_rate = model.generation_config.sample_rate

preload_models()


def timer(func):  # @timer
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        if debug:
            print(f"Execution time of {func.__name__}: {end - start} seconds")
        return result
    return wrapper


def readfile(file="uid.txt", mod="r", cont=None, jso: bool = False):
    if not mod in ("w", "a", ):
        assert os.path.isfile(file), str(file)
    if mod == "r":
        with open(file, encoding="utf-8") as file:
            lines: list = file.readlines()
        return lines
    elif mod == "_r":
        with open(file, encoding="utf-8") as file:
            contents = file.read() if not jso else json.load(file)
        return contents
    elif mod == "rb":
        with open(file, mod) as file:
            contents = file.read()
        return contents
    elif mod in ("w", "a", ):
        with open(file, mod, encoding="utf-8") as fil_e:
            if not jso:
                fil_e.write(cont)
            else:
                json.dump(cont, fil_e, indent=2, ensure_ascii=False)


def Py_Transformers_(aud___in, voice_preset, length_penalty=1.):
    print(f"*{aud___in}")
    inputs = processor(
        aud___in,
        # voice_preset=voice_preset
    )
    audio_array = model.generate(
        **inputs.to(device),
        # length_penalty=length_penalty,
        # num_beams=4,
        # # temperature=0.5,
        # # semantic_temperature=0.8,
    )
    return audio_array.cpu().numpy().squeeze()


def Py_Bark_(aud___in, voice_preset):
    print(f"#{aud___in}")
    return generate_audio(
        aud___in,
        # text_temp=0.3,
        # waveform_temp=0.3,
        # output_full=True,  # AttributeError: 'tuple' object has no attribute 'dtype'
        # history_prompt=voice_preset
    )


def Py_Transformers(aud___in, voice_preset, length_penalty=1.):
    silence = np.zeros(int(0.25 * SAMPLE_RATE))
    sentences = nltk.sent_tokenize(aud___in.replace('♪', ''))

    pieces = []
    for sentence in sentences:
        sentence = f'♪ {sentence} ♪'
        print(f'#{sentence}')
        inputs = processor(sentence)
        audio_array = model.generate(
            **inputs.to(device),
        )
        pieces += [audio_array.cpu().numpy().squeeze(), silence.copy()]
    return np.concatenate(pieces)


def Py_Bark(aud___in, voice_preset):
    silence = np.zeros(int(0.25 * SAMPLE_RATE))
    sentences = nltk.sent_tokenize(aud___in.replace('♪', ''))

    pieces = []
    for sentence in sentences:
        sentence = f'♪ {sentence} ♪'
        print(f'#{sentence}')
        audio_array = generate_audio(
            sentence,
        )
        pieces += [audio_array, silence.copy()]
    return np.concatenate(pieces)


if __name__ == '__main__':
    from scipy.io.wavfile import write as write_wav

    for sen in (
            "♪one two three four five six lalala♪",
            "♪ one two three four five six lalala ♪",
            "♪  one two three four five six lalala  ♪",

            "♪ I'm a Barbie girl, in the Barbie world ♪",
            "♪ Life in plastic, it's fantastic. ♪",
            "♪You can brush my hair, undress me everywhere.♪",
            "♪ Imagination, life is your creation. ♪",
    ):
        for i in range(3):
            audio_array = Py_Bark(sen, "v2/en_speaker_5")
            write_wav(f"{sen}__________{i}.wav", SAMPLE_RATE, audio_array)
