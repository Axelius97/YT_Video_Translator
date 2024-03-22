import streamlit as st
import spacy
import os  # Import the os module
from youtube_transcript_api import YouTubeTranscriptApi
from langchain import OpenAI, PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.chat_models import ChatOpenAI
import pytube
from moviepy.editor import VideoFileClip, AudioFileClip
from gtts import gTTS
from io import StringIO
import PyPDF2
import base64
from transformers import pipeline

nlp = spacy.load("en_core_web_sm")

global transcript
transcript = ''

st.set_page_config(
    page_title="Kalpana Project Phase",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

language_codes = {'Hindi': 'hi', 'Malayalam': 'ml', 'Tamil': 'ta', 'Telugu': 'te', 'Kannada': 'kn', 'Bengali': 'bn', 'Gujarati': 'gu', 'Marathi': 'mr'}

# Access the OpenAI API key from environment variable
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

def convert_func():
    if ytlink:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(ytlink, languages=['en', 'ar', 'en-IN', 'en-GB', 'en-US', 'en-AU', 'en-CA', 'en-IE', 'en-ZA', 'en-JM', 'en-NZ', 'en-PH', 'en-TT', 'en-ZW'])
        except Exception as e: 
            st.error('Error in fetching transcript')
            st.error(e)
        
        original_transcript = ''

        for i in transcript:
            original_transcript += i['text'] + ' '

        st.markdown(f"<h3 style='text-align: center; color:pink'>ORIGINAL TRANSCRIPT</h3>", unsafe_allow_html=True)
        st.write(original_transcript)

        st.write('')

        ans = convert_lang(original_transcript, target_language)
        st.markdown(f"<h3 style='text-align: center; color:pink'>CONVERTED TRANSCRIPT</h3>", unsafe_allow_html=True)
        st.write(ans)

        url = f"https://www.youtube.com/watch?v={ytlink}"

        yt = pytube.YouTube(url)
        video = yt.streams.filter(progressive=True, file_extension="mp4").first()

        custom_filename = f"{ytlink}.mp4"
        video.download(filename=custom_filename)

        video_clip = VideoFileClip(custom_filename)
        mute_and_play_text_over_video(custom_filename, ans)
            
    elif transcript1:
        binary_data = transcript1.getvalue()
        pdfReader = PyPDF2.PdfReader(transcript1)

        text = ''
        for i in range(0, len(pdfReader.pages)):
            pageObj = pdfReader.pages[i]
            text += pageObj.extract_text() + ' '

        ans2 = convert_lang(text, language_codes[target_language])
        tts = gTTS(text=ans2, lang=language_codes[target_language])
        tts.save("speech_target.mp3")
        audio_file = open('speech_target.mp3', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3')
        
        base64_pdf = base64.b64encode(binary_data).decode('utf-8')

        pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="800" type="application/pdf"></iframe>'

        st.markdown(pdf_display, unsafe_allow_html=True)

def mute_and_play_text_over_video(ytlink, text):
    url = f"https://www.youtube.com/watch?v={ytlink}"

    yt = pytube.YouTube(url)
    video = yt.streams.filter(progressive=True, file_extension="mp4").first()

    custom_filename = f"{ytlink}.mp4"
    video.download(filename=custom_filename)

    video_clip = VideoFileClip(custom_filename)
    muted_clip = video_clip.without_audio()

    tts = gTTS(text=text, lang='en')  
    tts.save("tts.mp3")

    final_clip = muted_clip.set_audio(AudioFileClip("tts.mp3"))

    final_clip.write_videofile(f"muted_with_tts_{custom_filename}")

    video_file = open(f'muted_with_tts_{ytlink}.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

def convert_lang(old_text, target_language):
    # Use the OpenAI API key from environment variable
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=.9, openai_api_key=OPENAI_API_KEY)
    prompt1 = PromptTemplate(
    template = """
                You are given text : {texty}.\n
                You are given the target language : {target_lang}.\n
                You have to convert text to target language
        """, input_variables=["texty", "target_lang"]
    )

    prompt_chain1 = LLMChain(llm=llm, prompt=prompt1)
    LLM_chain = SequentialChain(chains=[prompt_chain1], input_variables = ["texty", "target_lang"], verbose=True)

    tar_lang = LLM_chain({"texty":old_text, "target_lang":target_language})

    output_lang = tar_lang['text']

    return output_lang

st.title("Welcome to our App")

choice = st.sidebar.selectbox('choose one', ['YouTube Langauge converter', 'Q n a sys'])

if choice == "YouTube Langauge converter":
    ytlink = st.text_input('Enter the youtube Video ID')
    st.text('OR')
    transcript1 = st.file_uploader('Upload a file...', type="pdf")
    target_language = st.selectbox('Select Language', ['Hindi', 'Malayalam', 'Tamil', 'Telugu', 'Kannada', 'Bengali', 'Gujarati', 'Marathi'])
    convert = st.button('Convert')

    if convert:
        convert_func()

else:
    qna = st.text_input('Enter the youtube Video ID')
    if qna:
        try:
            transcript2 = YouTubeTranscriptApi.get_transcript(qna, languages=['en', 'ar', 'en-IN', 'en-GB', 'en-US', 'en-AU', 'en-CA', 'en-IE', 'en-ZA', 'en-JM', 'en-NZ', 'en-PH', 'en-TT', 'en-ZW'])
        except Exception as e: 
            st.error('Error in fetching transcript')
            st.error(e)

        original_transcript2 = ''

        for i in transcript2:
            original_transcript2 += i['text'] + ' '

        user_input = st.text_input('Enter Your Question')

        model_name = "deepset/roberta-base-squad2"

        nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
        if user_input:
            QA_input = {
                'question': user_input,
                'context':   original_transcript2
            }
            res = nlp(QA_input)
            st.write(res['answer'])
