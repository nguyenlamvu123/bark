import base64, os, scipy
import streamlit as st

from coordinate_constant import streamlit, historyfile, sample_rate, timer, \
    readfile, Py_Transformers, Py_Bark
from coordinate_constant import temp as te_mp


def main_loop_strl():
    def rendhtmlaudio():
        html: str = ''
        histlist: list = readfile(file=historyfile)
        for ih in range(0, len(histlist), 2):
            out___mp4, b64 = histlist[ih], histlist[ih + 1]
            html += f"""<h5>{out___mp4}</h5>
            <audio controls>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            <a download={out___mp4} href="data:audio/mp3;base64,{b64}">Download</a>
            """
        return html

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
            aud___in = f'♪ {aud___in} ♪'
            submit = st.form_submit_button('Chạy!')  # https://blog.streamlit.io/introducing-submit-button-and-forms/

    if not submit:
        return None
    main_loop(aud___in, genre)


@timer
def main_loop(aud___in: str, genre, voice_preset: str = "v2/en_speaker_5", length_penalty=1., out___mp4_=None):
    placeholder = st.empty() if streamlit else None

    st.write(aud___in)
    audio_array = Py_Transformers(aud___in, voice_preset, length_penalty=length_penalty) if genre == "Python Transformers" \
        else Py_Bark(aud___in, voice_preset)

    if out___mp4_ is None: out___mp4_ = f"{aud___in[:30].replace(' ', '')}.wav"
    scipy.io.wavfile.write(out___mp4_, rate=sample_rate, data=audio_array)
    if streamlit:
        data = readfile(file=out___mp4_, mod="rb")
        st.audio(audio_array, sample_rate=sample_rate, format='wav')
        # st.audio(data, format='wav')
        b64 = base64.b64encode(data).decode()
        readfile(file=historyfile, mod="a", cont=f'{aud___in}: \n{b64}\n')  # ghi lại lịch sử dưới dạng base64 vào file trên local
        with placeholder.container():
            st.write(aud___in)
            st.download_button(
                label="Download",
                data=data,
                file_name=out___mp4_,
                mime='wav',
            )


if __name__ == '__main__':
    main_loop_strl()  # streamlit run strlit.py --server.port 8501
    # python3 manage.py runserver 0.0.0.0:8501  # TODO
