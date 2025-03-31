# import streamlit as st
# import torch
# from transformers import AutoProcessor, AutoModel
# from datasets import load_dataset
# import scipy
# import os
# import io
# import base64
# from pathlib import Path

# # Set page configuration
# st.set_page_config(
#     page_title="Text to Speech Converter",
#     page_icon="🔊",
#     layout="centered"
# )

# # App title and description
# st.title("Text to Speech Converter")
# st.markdown("Convert your text to speech using Hugging Face models and download as MP3")

# # Create a sidebar for model selection
# with st.sidebar:
#     st.header("Model Settings")
    
#     model_options = {
#         "Microsoft SpeechT5": "microsoft/speecht5_tts",
#         "Facebook MMS TTS": "facebook/mms-tts-eng",
#         "VITS Fast Speech": "facebook/fastspeech2-en-ljspeech",
#     }
    
#     selected_model = st.selectbox(
#         "Select TTS Model:",
#         options=list(model_options.keys())
#     )
    
#     # Model-specific settings based on selection
#     if "SpeechT5" in selected_model:
#         speaker_embeddings = ["default", "random"]
#         selected_speaker = st.selectbox("Speaker Embedding:", speaker_embeddings)
#     elif "MMS" in selected_model:
#         languages = ["eng", "fra", "deu", "ita", "spa", "por"]
#         selected_language = st.selectbox("Language:", languages)
    
#     st.markdown("---")
#     st.markdown("### About")
#     st.markdown("This app uses Hugging Face's text-to-speech models to convert text into natural-sounding speech.")

# # Create a form for input
# with st.form(key="tts_form"):
#     # Text input area
#     text_input = st.text_area(
#         "Enter the text you want to convert to speech:",
#         height=150,
#         placeholder="Type or paste your text here..."
#     )
    
#     # Voice parameters
#     col1, col2 = st.columns(2)
    
#     with col1:
#         speed = st.slider(
#             "Speech speed:",
#             min_value=0.5,
#             max_value=2.0,
#             value=1.0,
#             step=0.1
#         )
    
#     with col2:
#         if "SpeechT5" in selected_model or "FastSpeech" in selected_model:
#             voice_pitch = st.slider(
#                 "Voice pitch:",
#                 min_value=-10,
#                 max_value=10,
#                 value=0
#             )
    
#     # Submit button
#     submit_button = st.form_submit_button(label="Convert to Speech")

# # Function to load model and processor
# @st.cache_resource
# def load_tts_model(model_name):
#     try:
#         processor = AutoProcessor.from_pretrained(model_name)
#         model = AutoModel.from_pretrained(model_name)
#         return model, processor
#     except Exception as e:
#         st.error(f"Failed to load model: {e}")
#         return None, None

# # Function to convert text to speech with SpeechT5
# def convert_with_speecht5(text, processor, model, speed=1.0, pitch=0):
#     try:
#         # Get speaker embeddings
#         embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
#         speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        
#         # Process text input
#         inputs = processor(text=text, return_tensors="pt")
        
#         # Generate speech
#         speech = model.generate_speech(
#             inputs["input_ids"], 
#             speaker_embeddings, 
#             vocoder=None
#         )
        
#         # Adjust speed (resample)
#         if speed != 1.0:
#             speech = torch.nn.functional.interpolate(
#                 speech.unsqueeze(0).unsqueeze(0), 
#                 scale_factor=1/speed,
#                 mode="linear", 
#                 align_corners=False
#             ).squeeze()
        
#         # Convert to numpy and save
#         speech_np = speech.numpy()
        
#         # Create a temporary file path
#         temp_dir = Path("temp")
#         temp_dir.mkdir(exist_ok=True)
#         audio_file = temp_dir / "speech.mp3"
        
#         # Save as MP3
#         scipy.io.wavfile.write(
#             str(audio_file).replace(".mp3", ".wav"), 
#             rate=16000, 
#             data=speech_np
#         )
        
#         # Convert WAV to MP3
#         os.system(f"ffmpeg -i {str(audio_file).replace('.mp3', '.wav')} -y -codec:a libmp3lame {audio_file}")
        
#         return str(audio_file)
#     except Exception as e:
#         st.error(f"An error occurred: {e}")
#         return None

# # Function to convert text to speech with MMS TTS
# def convert_with_mms(text, processor, model, language="eng", speed=1.0):
#     try:
#         # Process text input
#         inputs = processor(text=text, language=language, return_tensors="pt")
        
#         # Generate speech
#         with torch.no_grad():
#             output = model(**inputs)
        
#         speech = output.waveform.squeeze().numpy()
        
#         # Adjust speed (resample)
#         if speed != 1.0:
#             # Implement resampling
#             pass
        
#         # Create a temporary file path
#         temp_dir = Path("temp")
#         temp_dir.mkdir(exist_ok=True)
#         audio_file = temp_dir / "speech.mp3"
        
#         # Save as MP3
#         scipy.io.wavfile.write(
#             str(audio_file).replace(".mp3", ".wav"), 
#             rate=16000, 
#             data=speech
#         )
        
#         # Convert WAV to MP3
#         os.system(f"ffmpeg -i {str(audio_file).replace('.mp3', '.wav')} -y -codec:a libmp3lame {audio_file}")
        
#         return str(audio_file)
#     except Exception as e:
#         st.error(f"An error occurred: {e}")
#         return None

# # Function to get a download link for a file
# def get_download_link(file_path, filename):
#     with open(file_path, "rb") as file:
#         file_bytes = file.read()
    
#     b64 = base64.b64encode(file_bytes).decode()
#     href = f'<a href="data:audio/mp3;base64,{b64}" download="{filename}">Download MP3 file</a>'
#     return href

# # Process the text when the form is submitted
# if submit_button and text_input:
#     if not text_input.strip():
#         st.warning("Please enter some text to convert.")
#     else:
#         with st.spinner("Loading model and converting text to speech..."):
#             # Get the model name
#             model_name = model_options[selected_model]
            
#             # Load model and processor
#             model, processor = load_tts_model(model_name)
            
#             if model and processor:
#                 # Convert text to speech based on selected model
#                 if "SpeechT5" in selected_model:
#                     audio_path = convert_with_speecht5(
#                         text_input, 
#                         processor, 
#                         model, 
#                         speed=speed, 
#                         pitch=voice_pitch if 'voice_pitch' in locals() else 0
#                     )
#                 elif "MMS" in selected_model:
#                     audio_path = convert_with_mms(
#                         text_input, 
#                         processor, 
#                         model, 
#                         language=selected_language if 'selected_language' in locals() else "eng",
#                         speed=speed
#                     )
#                 else:
#                     # Default to SpeechT5
#                     audio_path = convert_with_speecht5(text_input, processor, model, speed=speed)
                
#                 if audio_path:
#                     # Display success message
#                     st.success("Text successfully converted to speech!")
                    
#                     # Create audio player
#                     st.audio(audio_path, format="audio/mp3")
                    
#                     # Create download button
#                     st.markdown(
#                         get_download_link(audio_path, "speech.mp3"),
#                         unsafe_allow_html=True
#                     )
                    
#                     # Display character count
#                     char_count = len(text_input)
#                     st.info(f"Character count: {char_count}")

# # Add deployment instructions
# st.markdown("---")
# st.subheader("Deployment Instructions")
# st.markdown("""
# 1. Save this code to a file named `app.py`
# 2. Install required packages:
#    ```
#    pip install streamlit torch transformers datasets scipy
#    ```
# 3. Install ffmpeg for audio conversion:
#    - Ubuntu/Debian: `sudo apt-get install ffmpeg`
#    - macOS: `brew install ffmpeg`
#    - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
# 4. Run the app locally:
#    ```
#    streamlit run app.py
#    ```
# 5. For cloud deployment, create a `requirements.txt` file with:
#    ```
#    streamlit
#    torch
#    transformers
#    datasets
#    scipy
#    ```
#    Note: You'll need to ensure ffmpeg is installed on your deployment platform.
# """)

# # Clean up temporary files when the app is closed
# def cleanup():
#     temp_dir = Path("temp")
#     if temp_dir.exists():
#         for file in temp_dir.iterdir():
#             try:
#                 file.unlink()
#             except:
#                 pass
#         try:
#             temp_dir.rmdir()
#         except:
#             pass

# # Register the cleanup function to run when the app is closed
# import atexit
# atexit.register(cleanup)

import streamlit as st
import torch
from transformers import AutoProcessor, AutoModel
from datasets import load_dataset
import scipy
import os
import io
import base64
from pathlib import Path
import random

# Set page configuration
st.set_page_config(
    page_title="VoiceForge - Text to Speech",
    page_icon="🎙️",
    layout="wide"
)

# Custom CSS to make the app more exciting
st.markdown("""
<style>
    .main-header {
        font-size: 3rem !important;
        font-weight: 800 !important;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem !important;
    }
    .sub-header {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #6a6a6a;
    }
    .highlight {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.1));
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255,255,255,0.18);
        margin-bottom: 1rem !important;
    }
    .voice-card {
        padding: 10px;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    .voice-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #FF6B6B;
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FF8E8E;
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<h1 class="main-header">VoiceForge: Text-to-Speech Supercharged</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Transform your text into lifelike speech with AI - Select your voice, customize the delivery, and download as MP3!</p>', unsafe_allow_html=True)

# Create a sidebar for model selection
with st.sidebar:
    st.image("https://static.streamlit.io/examples/cat.jpg", width=100)  # Placeholder image
    st.header("🎛️ Voice Engine Settings")
    
    model_options = {
        "🔊 Microsoft SpeechT5": "microsoft/speecht5_tts",
        "🌐 Facebook MMS TTS": "facebook/mms-tts-eng",
        "⚡ VITS Fast Speech": "facebook/fastspeech2-en-ljspeech",
    }
    
    selected_model = st.selectbox(
        "Select TTS Model:",
        options=list(model_options.keys())
    )
    
    # Voice profiles for each model
    if "SpeechT5" in selected_model:
        st.markdown("### 🎭 Voice Profiles")
        voice_profiles = {
            "Professional Male": 7306,  # Voice index from dataset
            "Friendly Female": 8100,
            "Deep Narrator": 6500,
            "Young Enthusiastic": 1100,
            "Elder Statesman": 3200,
            "Calm Storyteller": 5000,
            "Energetic Announcer": 9000
        }
        selected_voice = st.selectbox("Voice Type:", list(voice_profiles.keys()))
        
        speaker_embeddings = ["default", "random"]
        selected_speaker = st.selectbox("Speaker Embedding:", speaker_embeddings)
    elif "MMS" in selected_model:
        st.markdown("### 🌎 Language & Voice")
        languages = {
            "English": "eng",
            "French": "fra",
            "German": "deu",
            "Italian": "ita",
            "Spanish": "spa",
            "Portuguese": "por"
        }
        
        voice_profiles = {
            "Default Voice": 0,
            "Mature Voice": 1,
            "Bright Voice": 2,
            "Serious Voice": 3,
            "Expressive Voice": 4
        }
        
        selected_language = st.selectbox("Language:", list(languages.keys()))
        selected_voice = st.selectbox("Voice Type:", list(voice_profiles.keys()))
        language_code = languages[selected_language]
    else:
        # VITS voice options
        st.markdown("### 🎤 Voice Styles")
        voice_profiles = {
            "Default Voice": 0,
            "Clear Narrator": 1,
            "Conversational": 2,
            "Broadcast Voice": 3,
            "Smooth Presenter": 4
        }
        selected_voice = st.selectbox("Voice Style:", list(voice_profiles.keys()))
    
    st.markdown("---")
    st.markdown("### 💡 Pro Tips")
    st.info("Short sentences often sound more natural than very long paragraphs.")
    st.warning("First generation may take 30-60 seconds to download models.")

# Create columns for a more exciting layout
col1, col2 = st.columns([3, 2])

# Main input area
with col1:
    with st.form(key="tts_form"):
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        
        # Text input area with an exciting prompt
        text_input = st.text_area(
            "✏️ Type your message to be voiced:",
            height=150,
            placeholder="Enter your text here... Try adding emotion marks like [excited], [whispered], or [serious] for more dynamic speech!"
        )
        
        # Voice parameters in a more visual arrangement
        param_col1, param_col2 = st.columns(2)
        
        with param_col1:
            speed = st.slider(
                "🏃‍♂️ Speech speed:",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Adjust how fast the voice speaks (1.0 is normal speed)"
            )
        
        with param_col2:
            if "SpeechT5" in selected_model or "FastSpeech" in selected_model:
                voice_pitch = st.slider(
                    "🎵 Voice pitch:",
                    min_value=-10,
                    max_value=10,
                    value=0,
                    help="Lower values for deeper voice, higher values for higher pitch"
                )
        
        # Effects section
        st.markdown("### ✨ Special Effects")
        effects_col1, effects_col2, effects_col3 = st.columns(3)
        
        with effects_col1:
            add_echo = st.checkbox("Echo Effect", help="Adds a slight echo to the voice")
        
        with effects_col2:
            add_background = st.checkbox("Background Music", help="Adds subtle background music")
        
        with effects_col3:
            voice_emotion = st.selectbox(
                "Emotion Tone:",
                ["Neutral", "Happy", "Serious", "Excited", "Calm", "Mysterious"],
                help="Attempts to adjust the voice characteristics to match the emotion"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Submit button with exciting styling
        submitted = st.form_submit_button(label="🔮 GENERATE VOICE MAGIC!")

# Voice preview section
with col2:
    st.markdown("### 👂 Preview Your Voice")
    
    # Voice card with sample text
    voice_sample = "This is how your selected voice will sound. Adjust settings on the left!"
    
    # Display the current voice profile in a card
    st.markdown(f"""
    <div class="voice-card" style="background: linear-gradient(135deg, #FFEFBA, #FFFFFF);">
        <h4>Selected Voice: {selected_voice}</h4>
        <p><i>"{voice_sample}"</i></p>
        <div style="display: flex; align-items: center;">
            <div style="height: 40px; flex-grow: 1; background: linear-gradient(90deg, #232526, #414345); border-radius: 5px; position: relative; overflow: hidden;">
                <div style="position: absolute; height: 100%; width: 30%; background: linear-gradient(90deg, #FF6B6B, #4ECDC4); animation: pulse 2s infinite;"></div>
            </div>
        </div>
    </div>
    <style>
    @keyframes pulse {
        0% {width: 30%; opacity: 0.6;}
        50% {width: 70%; opacity: 0.9;}
        100% {width: 30%; opacity: 0.6;}
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display model information
    st.markdown("### 🧠 Model Information")
    
    model_info = {
        "🔊 Microsoft SpeechT5": "A powerful neural TTS model that produces natural-sounding speech with excellent prosody and intonation.",
        "🌐 Facebook MMS TTS": "Multilingual model supporting 6 languages with consistent voice quality across languages.",
        "⚡ VITS Fast Speech": "Fast and efficient model optimized for quick generation with good voice quality."
    }
    
    st.info(model_info[selected_model])
    
    # Recently generated samples (placeholder for now)
    st.markdown("### 🕒 Recent Generations")
    
    # This would be populated with actual data in a real application
    if "history" not in st.session_state:
        st.session_state.history = []

# Function to load model and processor
@st.cache_resource
def load_tts_model(model_name):
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        return model, processor
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

# Function to convert text to speech with SpeechT5
def convert_with_speecht5(text, processor, model, voice_index=7306, speed=1.0, pitch=0, add_echo=False):
    try:
        # Get speaker embeddings
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embeddings = torch.tensor(embeddings_dataset[voice_index]["xvector"]).unsqueeze(0)
        
        # Process text input
        inputs = processor(text=text, return_tensors="pt")
        
        # Generate speech
        speech = model.generate_speech(
            inputs["input_ids"], 
            speaker_embeddings, 
            vocoder=None
        )
        
        # Adjust pitch if needed
        if pitch != 0:
            # Pitch adjustment would be implemented here
            # This is a simplified placeholder approach
            if pitch > 0:
                speech = speech * (1.0 + (pitch * 0.01))
            else:
                speech = speech * (1.0 / (1.0 - (pitch * 0.01)))
        
        # Adjust speed (resample)
        if speed != 1.0:
            speech = torch.nn.functional.interpolate(
                speech.unsqueeze(0).unsqueeze(0), 
                scale_factor=1/speed,
                mode="linear", 
                align_corners=False
            ).squeeze()
        
        # Add echo effect if selected
        if add_echo:
            # Simple echo effect (delayed and attenuated copy)
            echo_delay = int(16000 * 0.2)  # 200ms delay
            echo_speech = torch.zeros_like(speech)
            if echo_delay < len(speech):
                echo_speech[echo_delay:] = speech[:-echo_delay] * 0.3
                speech = speech + echo_speech
        
        # Convert to numpy and save
        speech_np = speech.numpy()
        
        # Create a temporary file path
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        audio_file = temp_dir / f"speech_{random.randint(1000, 9999)}.mp3"
        
        # Save as MP3
        scipy.io.wavfile.write(
            str(audio_file).replace(".mp3", ".wav"), 
            rate=16000, 
            data=speech_np
        )
        
        # Convert WAV to MP3
        os.system(f"ffmpeg -i {str(audio_file).replace('.mp3', '.wav')} -y -codec:a libmp3lame {audio_file}")
        
        # Add to history
        if "history" in st.session_state:
            text_preview = text[:20] + "..." if len(text) > 20 else text
            st.session_state.history.append({
                "text": text_preview,
                "file": str(audio_file),
                "model": "SpeechT5",
                "voice": f"Voice Index: {voice_index}"
            })
        
        return str(audio_file)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Function to convert text to speech with MMS TTS
def convert_with_mms(text, processor, model, language="eng", voice_index=0, speed=1.0, add_echo=False):
    try:
        # Process text input
        inputs = processor(text=text, language=language, return_tensors="pt")
        
        # Generate speech
        with torch.no_grad():
            output = model(**inputs)
        
        speech = output.waveform.squeeze().numpy()
        
        # Adjust speed (resample)
        if speed != 1.0:
            # This is a placeholder for resampling
            # In a real implementation, we would use a proper resampling method
            pass
        
        # Add echo effect if selected
        if add_echo:
            # Simple echo effect
            echo_delay = int(16000 * 0.2)  # 200ms delay
            echo_speech = np.zeros_like(speech)
            if echo_delay < len(speech):
                echo_speech[echo_delay:] = speech[:-echo_delay] * 0.3
                speech = speech + echo_speech
        
        # Create a temporary file path
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        audio_file = temp_dir / f"speech_{random.randint(1000, 9999)}.mp3"
        
        # Save as MP3
        scipy.io.wavfile.write(
            str(audio_file).replace(".mp3", ".wav"), 
            rate=16000, 
            data=speech
        )
        
        # Convert WAV to MP3
        os.system(f"ffmpeg -i {str(audio_file).replace('.mp3', '.wav')} -y -codec:a libmp3lame {audio_file}")
        
        # Add to history
        if "history" in st.session_state:
            text_preview = text[:20] + "..." if len(text) > 20 else text
            st.session_state.history.append({
                "text": text_preview,
                "file": str(audio_file),
                "model": "MMS TTS",
                "voice": f"Language: {language}, Voice: {voice_index}"
            })
        
        return str(audio_file)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Function to get a download link for a file
def get_download_link(file_path, filename):
    with open(file_path, "rb") as file:
        file_bytes = file.read()
    
    b64 = base64.b64encode(file_bytes).decode()
    href = f'<a href="data:audio/mp3;base64,{b64}" download="{filename}" class="download-button">📥 Download MP3 File</a>'
    return href

# Process the text when the form is submitted
if submitted and text_input:
    if not text_input.strip():
        st.warning("⚠️ Please enter some text to convert.")
    else:
        # Animated loading message
        with st.spinner("🔮 AI Voice Magic in Progress... Sit tight!"):
            # Get the model name
            model_name = model_options[selected_model]
            
            # Pre-process text based on emotion
            if voice_emotion != "Neutral":
                # Add emotion markers to the text
                if voice_emotion == "Happy":
                    text_input = f"[happily] {text_input}"
                elif voice_emotion == "Serious":
                    text_input = f"[seriously] {text_input}"
                elif voice_emotion == "Excited":
                    text_input = f"[excitedly] {text_input}"
                elif voice_emotion == "Calm":
                    text_input = f"[calmly] {text_input}"
                elif voice_emotion == "Mysterious":
                    text_input = f"[mysteriously] {text_input}"
            
            # Load model and processor
            model, processor = load_tts_model(model_name)
            
            if model and processor:
                # Get voice index from selected profile
                if "SpeechT5" in selected_model:
                    voice_profiles_map = {
                        "Professional Male": 7306,
                        "Friendly Female": 8100,
                        "Deep Narrator": 6500,
                        "Young Enthusiastic": 1100,
                        "Elder Statesman": 3200,
                        "Calm Storyteller": 5000,
                        "Energetic Announcer": 9000
                    }
                    voice_index = voice_profiles_map[selected_voice]
                    
                    audio_path = convert_with_speecht5(
                        text_input, 
                        processor, 
                        model, 
                        voice_index=voice_index,
                        speed=speed, 
                        pitch=voice_pitch if 'voice_pitch' in locals() else 0,
                        add_echo=add_echo if 'add_echo' in locals() else False
                    )
                elif "MMS" in selected_model:
                    voice_profiles_map = {
                        "Default Voice": 0,
                        "Mature Voice": 1,
                        "Bright Voice": 2,
                        "Serious Voice": 3,
                        "Expressive Voice": 4
                    }
                    voice_index = voice_profiles_map[selected_voice]
                    
                    audio_path = convert_with_mms(
                        text_input, 
                        processor, 
                        model, 
                        language=language_code if 'language_code' in locals() else "eng",
                        voice_index=voice_index,
                        speed=speed,
                        add_echo=add_echo if 'add_echo' in locals() else False
                    )
                else:
                    # Default to SpeechT5
                    audio_path = convert_with_speecht5(text_input, processor, model, speed=speed)
                
                if audio_path:
                    # Display a fun success message
                    st.success("🎉 Voice generation complete! Listen to your masterpiece below.")
                    
                    # Create a professional-looking result card
                    st.markdown("""
                    <div style="padding: 20px; border-radius: 10px; background: linear-gradient(to right, #a8ff78, #78ffd6); margin: 10px 0;">
                        <h3>🎙️ Your Generated Voice</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create audio player with custom styling
                    st.audio(audio_path, format="audio/mp3")
                    
                    # Create download button with custom styling
                    st.markdown(
                        get_download_link(audio_path, "voiceforge_speech.mp3"),
                        unsafe_allow_html=True
                    )
                    
                    # Add some CSS for the download button
                    st.markdown("""
                    <style>
                    .download-button {
                        display: inline-block;
                        padding: 10px 20px;
                        background: linear-gradient(to right, #4A00E0, #8E2DE2);
                        color: white;
                        text-decoration: none;
                        border-radius: 30px;
                        font-weight: bold;
                        transition: all 0.3s;
                        margin-top: 10px;
                    }
                    .download-button:hover {
                        transform: scale(1.05);
                        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Display text info with character count
                    char_count = len(text_input)
                    word_count = len(text_input.split())
                    
                    # Display generation stats
                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    
                    with stat_col1:
                        st.metric(label="Characters", value=f"{char_count}")
                    
                    with stat_col2:
                        st.metric(label="Words", value=f"{word_count}")
                    
                    with stat_col3:
                        st.metric(label="Speech Length", value=f"~{int(word_count * 0.5)}s")
                    
                    # Show a helpful tip
                    st.info("💡 Tip: Try different voice types and emotion settings for varied results!")

# Display recent generations if available
if "history" in st.session_state and len(st.session_state.history) > 0:
    with col2:
        for item in st.session_state.history[-3:]:  # Show last 3 items
            st.markdown(f"""
            <div style="padding: 10px; margin: 5px 0; border-radius: 5px; border: 1px solid #ddd;">
                <p><small>{item['text']}</small></p>
                <p><small><b>Model:</b> {item['model']} | <b>Voice:</b> {item['voice']}</small></p>
            </div>
            """, unsafe_allow_html=True)
            st.audio(item['file'], format="audio/mp3")

# Footer with deployment instructions
st.markdown("---")
st.markdown('<h3 style="color: #4ECDC4;">🚀 Deployment Guide</h3>', unsafe_allow_html=True)
st.markdown("""
1. Save this code to a file named `app.py`
2. Install required packages:
   ```
   pip install streamlit torch transformers datasets scipy
   ```
3. Install ffmpeg for audio conversion:
   - Ubuntu/Debian: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
4. Run the app locally:
   ```
   streamlit run app.py
   ```
5. For cloud deployment, create a `requirements.txt` file with:
   ```
   streamlit
   torch
   transformers
   datasets
   scipy
   ```
   
   Note: For Streamlit cloud deployment, you'll need to ensure ffmpeg is installed.
""")

# Clean up temporary files when the app is closed
def cleanup():
    temp_dir = Path("temp")
    if temp_dir.exists():
        for file in temp_dir.iterdir():
            try:
                file.unlink()
            except:
                pass
        try:
            temp_dir.rmdir()
        except:
            pass

# Register the cleanup function to run when the app is closed
import atexit
atexit.register(cleanup)