import base64, os, scipy
import streamlit as st

from coordinate_constant import \
    streamlit, historyfile, sample_rate, timer, temperature, ignotuple, \
    readfile, Py_Transformers, Py_Bark, Py_genetext, remove_numerical_order
    # max_length, num_return_sequences, num_beams
from coordinate_constant import temp as te_mp


def main_loop_strl():
    def rendhtmlaudio():
        raise AssertionError("error")
        # html: str = ''
        # histlist: list = readfile(file=historyfile)
        # for ih in range(0, len(histlist), 2):
        #     out___mp4, b64 = histlist[ih], histlist[ih + 1]
        #     html += f"""<h5>{out___mp4}</h5>
        #     <audio controls>
        #         <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        #     </audio>
        #     <a download={out___mp4} href="data:audio/mp3;base64,{b64}">Download</a>
        #     """
        # return html

    def dehi():
        if os.path.isfile(historyfile):
            os.rename(historyfile, te_mp + historyfile)

    st.title("Sinh nhạc")
    # st.subheader("This app allows you to find threshold to convert color Image to binary Image!")
    # st.text("We use OpenCV and Streamlit for this demo")

    # horizontal and center radio buttons
    st.write(
        '<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>',
        unsafe_allow_html=True
    )
    st.write(
        '<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>',
        unsafe_allow_html=True
    )

    # Inject custom CSS to set the width of the sidebar
    st.write(
        """
        <style>
            section[data-testid="stSidebar"] {
                width: 1500px !important; # Set the width to your desired value
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # show at the end of page
    st.write(  # https://stackoverflow.com/questions/41732055/how-to-set-the-div-at-the-end-of-the-page
        """
        <style>
            .banner {
              width: 100%;
              height: 15%;
              position: fixed;
              bottom: 0;
              overflow:auto;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    st.sidebar.button('xóa lịch sử', on_click=dehi)
    try:
        md = rendhtmlaudio()
    except AssertionError:
        md = "Xin chào!"
    st.markdown(  # đọc và hiện lịch sử
        f'<div class="banner">{md}</div>',
        unsafe_allow_html=True,
    )

    aud___in: str = "Hanoi is nicest at the night"
    with col1:
        genre = st.radio(
            "chọn phương án", ["Python bark", "Python Transformers", ]
        )
        with st.form("checkboxes", clear_on_submit=True):
            aud___in = st.text_input("nhập text để sinh nhạc", aud___in)
            # aud___in = f'♪ {aud___in} ♪'
            submit = st.form_submit_button('Chạy!')  # https://blog.streamlit.io/introducing-submit-button-and-forms/
    with col2:
        # https://huggingface.co/docs/transformers/v4.27.2/en/generation_strategies
        max_length = st.slider("độ dài text được sinh ra", min_value=30, max_value=500, step=10, value=50)
        num_return_sequences = st.slider("số trường hợp khác nhau được trả về", min_value=1, max_value=5, value=1)  # the number of sequence candidates to return for each input. This options is only available for the decoding strategies that support multiple sequence candidates, e.g. variations of beam search and sampling. Decoding strategies like greedy search and contrastive search return a single output sequence.
        num_beams = st.slider("số giả thuyết đánh giá ở mỗi bước", min_value=1, max_value=5, value=1)  # by specifying a number of beams higher than 1, you are effectively switching from greedy search to beam search. This strategy evaluates several hypotheses at each time step and eventually chooses the hypothesis that has the overall highest probability for the entire sequence. This has the advantage of identifying high-probability sequences that start with a lower probability initial tokens and would’ve been ignored by the greedy search
    with col3:
        if genre == "Python bark":
            text_temp = st.slider("độ linh hoạt trong mã hóa văn bản", min_value=0.0, max_value=1.0, step=0.1, value=0.8)
            waveform_temp = st.slider("độ linh hoạt trong tạo dạng sóng", min_value=0.0, max_value=1.0, value=0.9)

    if not submit:
        return None
    main_loop(
        aud___in, genre, max_length=max_length, num_return_sequences=num_return_sequences, num_beams=num_beams,
        text_temp=text_temp, waveform_temp=waveform_temp
    )


@timer
def main_loop(
        aud___in_: str, genre, out___mp4_=None, max_length: int = 50, num_return_sequences: int = 1, num_beams: int = 1,
        text_temp: float = 0.8, waveform_temp: float = 0.9,
):
    if streamlit:
        st.write(aud___in_)
        placeholder = st.empty()
    else:
        print(aud___in_)
        placeholder = None
    output_s: list = Py_genetext(
        "generate lyric about " + aud___in_,
        temperature=temperature,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        num_beams=num_beams
    )
    output_ret = list()
    for i, output in enumerate(output_s):
        assert isinstance(output, dict)
        assert 'generated_text' in output
        assert aud___in_ in output['generated_text']
        aud___in: str = '. '.join([s.strip(',') for s in output['generated_text'].split('\n')[1:] if all([
            not s.strip() == '',
            'Verse' not in s,
            'Chorus' not in s,
            not any([
                all([
                    s.startswith(ignotuple[z]),
                    s.endswith(f'{ignotuple[z + 1]}.'),
                ]) for z in range(0, len(ignotuple) - 1, 2)
            ]),
        ])])
        aud___in = remove_numerical_order(aud___in)
        print(f'{i}______{aud___in}')
        output_ret.append(aud___in)
        audio_array = Py_Transformers(aud___in) if genre == "Python Transformers" \
            else Py_Bark(aud___in, text_temp, waveform_temp)

        if out___mp4_ is None: out___mp4_ = f"{aud___in[:30].replace(' ', '')}_{i}.wav"
        scipy.io.wavfile.write(out___mp4_, rate=sample_rate, data=audio_array)
        if streamlit:
            # data = readfile(file=out___mp4_, mod="rb")
            st.write(aud___in)
            st.audio(audio_array, sample_rate=sample_rate, format='wav')
            # st.audio(data, format='wav')
            # b64 = base64.b64encode(data).decode()
            # readfile(file=historyfile, mod="a", cont=f'{aud___in}: \n{b64}\n')  # ghi lại lịch sử dưới dạng base64 vào file trên local
            # with placeholder.container():
            #     st.write(aud___in)
            #     st.download_button(
            #         label="Download",
            #         data=data,
            #         file_name=out___mp4_,
            #         mime='wav',
            #     )
    return output_ret


if __name__ == '__main__':
    main_loop_strl()  # streamlit run strlit.py --server.port 8501
    # python3 manage.py runserver 0.0.0.0:8501
