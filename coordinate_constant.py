import torch, json, os, requests, time

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
# default_cache_dir = os.path.join(os.path.expanduser("~"), ".cache")
CACHE_DIR = os.path.join(CUR_PATH, "bark", "loa_mod")
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['PYTORCH_PRETRAINED_BERT_CACHE'] = CACHE_DIR

from functools import wraps
from transformers import AutoProcessor, BarkModel, pipeline
from bark import SAMPLE_RATE, generate_audio, preload_models
import nltk  # we'll use this to split into sentences
import numpy as np


streamlit: bool = False
temp: str = 'temp'  # file temp ban đầu khi chuyển từng đoạn thoại ra audio
historyfile: str = "hist.txt"
debug: bool = True
temperature = 0.1
max_length = 100
num_return_sequences = 1
num_beams = 1
sen_except: list = [f'{s}.' for s in range(10)]
ignotuple = ('<', '>', '(', ')', '[', ']', '{', '}', )

device = "cuda:0" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")
# model = BarkModel.from_pretrained("suno/bark-small")
model = model.to(device)
sample_rate = model.generation_config.sample_rate
generator = pipeline(
    'text-generation',
    model='HuggingFaceH4/zephyr-7b-beta',
    model_kwargs={
        'cache_dir': CACHE_DIR,
    },
)
""" - `"audio-classification"`
    - `"automatic-speech-recognition"`
    - `"conversational"`
    - `"depth-estimation"`
    - `"document-question-answering"`
    - `"feature-extraction"`
    - `"fill-mask"`
    - `"image-classification"`
    - `"image-feature-extraction"`
    - `"image-segmentation"`
    - `"image-to-text"`
    - `"image-to-image"`
    - `"object-detection"`
    - `"question-answering"`
    - `"summarization"`
    - `"table-question-answering"`
    - `"text2text-generation"`
    - `"text-classification"` (alias `"sentiment-analysis"` available)
    - `"text-generation"`
    - `"text-to-audio"` (alias `"text-to-speech"` available)
    - `"token-classification"` (alias `"ner"` available)
    - `"translation"`
    - `"translation_xx_to_yy"`
    - `"video-classification"`
    - `"visual-question-answering"` (alias `"vqa"` available)
    - `"zero-shot-classification"`
    - `"zero-shot-image-classification"`
    - `"zero-shot-object-detection"`"""

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


def apply_nltk(func):  # @apply_nltk
    @wraps(func)
    def wrapper(*args, **kwargs):
        silence = np.zeros(int(0.25 * SAMPLE_RATE))
        # sentences = nltk.sent_tokenize(aud___in.replace('♪', ''))
        sentences = nltk.sent_tokenize(args[0])

        pieces = []
        for sentence in sentences:
            if sentence.strip() in sen_except:
                continue
            # sentence = f'♪ {sentence} ♪'
            # print(f'#{sentence}')
            audio_array = func(sentence)  # model.generate(**inputs.to(device))
            pieces += [audio_array, silence.copy()]
        return np.concatenate(pieces)
    return wrapper


@apply_nltk
@timer
def Py_Transformers_(aud___in, voice_preset=None, length_penalty=1.):
    print(f"*{aud___in}")
    inputs = processor(
        f'♪ {aud___in} ♪',
        # voice_preset=voice_preset
    )
    audio_array = model.generate(
        **inputs.to(device),
        # # length_penalty=length_penalty,
        # # num_beams=4,
        # temperature=0.1,
        # semantic_temperature=0.1,
    )
    return audio_array.cpu().numpy().squeeze()


@apply_nltk
@timer
def Py_Bark_(aud___in, voice_preset=None):
    print(f"#########{aud___in}")
    return generate_audio(
        f'♪ {aud___in} ♪',
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
        audio_array = model.generate(**inputs.to(device))
        pieces += [audio_array.cpu().numpy().squeeze(), silence.copy()]
    return np.concatenate(pieces)


def Py_Bark(aud___in, voice_preset):
    silence = np.zeros(int(0.25 * SAMPLE_RATE))
    sentences = nltk.sent_tokenize(aud___in.replace('♪', ''))

    pieces = []
    for sentence in sentences:
        sentence = f'♪ {sentence} ♪'
        print(f'#{sentence}')
        audio_array = generate_audio(sentence)
        pieces += [audio_array, silence.copy()]
    return np.concatenate(pieces)


def Py_genetext(
        aud___in,
        do_sample=True,
        top_p=0.95,
        temperature=0.8,
        max_length=50,
        num_return_sequences=4,
        num_beams=5,
        truncation=True
):
    return generator(
        aud___in,
        do_sample=do_sample,
        top_p=top_p,
        temperature=temperature,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        num_beams=num_beams,
        truncation=truncation
    )


if __name__ == '__main__':
    from scipy.io.wavfile import write as write_wav

    for sen in (
            # " how long have you troubled me, Huong Ly ",
            # "I'm a Barbie girl, in the Barbie world",
            # "Life in plastic, it's fantastic",
            "You can brush my hair, undress me everywhere",
            # "Imagination, life is your creation",
    ):
        output_s: list = Py_genetext(
            "generate lyric about " + sen,
            temperature=temperature,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            num_beams=num_beams
        )
        for i, output in enumerate(output_s):
            assert isinstance(output, dict)
            assert 'generated_text' in output
            assert sen in output['generated_text']
            aud___in: str = '. '.join([s.strip(',') for s in output['generated_text'].split('\n')[1:] if all([
                not s.strip() == '',
                'Verse' not in s,
                'Chorus:' not in s,
                not any([
                    all([
                        s.startswith(ignotuple[z]),
                        s.endswith(f'{ignotuple[z + 1]}.'),
                    ]) for z in range(0, len(ignotuple) - 1, 2)
                ]),
            ])])
            print(f'{i}______{aud___in}')
            audio_array = Py_Bark_(aud___in, "v2/en_speaker_5")
            write_wav(f"{i}.wav", SAMPLE_RATE, audio_array)
