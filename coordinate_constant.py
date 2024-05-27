import torch, os, json
from transformers import AutoProcessor, BarkModel
from bark import SAMPLE_RATE, generate_audio, preload_models


streamlit: bool = True
temp: str = 'temp'  # file temp ban đầu khi chuyển từng đoạn thoại ra audio
historyfile: str = "hist.txt"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")
# model = BarkModel.from_pretrained("suno/bark-small")
model = model.to(device)
sample_rate = model.generation_config.sample_rate

preload_models()


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


def Py_Transformers(aud___in, voice_preset):
    print("*")
    inputs = processor(
        aud___in,
        # voice_preset=voice_preset
    )
    audio_array = model.generate(
        **inputs.to(device),
        # num_beams=4,
        # temperature=0.5,
        # semantic_temperature=0.8,
    )
    return audio_array.cpu().numpy().squeeze()


def Py_Bark(aud___in, voice_preset):
    print("#")
    return generate_audio(
        aud___in,
        # text_temp=0.3,
        # waveform_temp=0.3,
        # output_full=True,  # AttributeError: 'tuple' object has no attribute 'dtype'
        # history_prompt=voice_preset
    )


if __name__ == '__main__':
    from scipy.io.wavfile import write as write_wav

    for sen in (
            "♪one two three four five six lalala♪",
            "♪ one two three four five six lalala ♪",
            "♪  one two three four five six lalala  ♪",
    ):
        for i in range(3):
            audio_array = Py_Bark(sen, "v2/en_speaker_5")
            write_wav(f"{sen}__________{i}.wav", SAMPLE_RATE, audio_array)
