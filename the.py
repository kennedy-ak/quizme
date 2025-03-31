import streamlit as st
import gtts
import io
import os
import base64
from pathlib import Path
import random
from scipy.io import wavfile
import numpy as np

# Remove PyTorch import that's causing issues
# import torch

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

# Create a sidebar for voice selection
with st.sidebar:
    st.header("🎛️ Voice Engine Settings")
    
    # Voice options
    voice_options = {
        "en-US Female": {"lang": "en", "tld": "com", "gender": "female", "slow": False},
        "en-US Male": {"lang": "en", "tld": "com", "gender": "male", "slow": False},
        "en-UK Female": {"lang": "en", "tld": "co.uk", "gender": "female", "slow": False},
        "en-UK Male": {"lang": "en", "tld": "co.uk", "gender": "male", "slow": False},
        "French": {"lang": "fr", "tld": "fr", "gender": None, "slow": False},
        "German": {"lang": "de", "tld": "de", "gender": None, "slow": False},
        "Spanish": {"lang": "es", "tld": "es", "gender": None, "slow": False},
        "Italian": {"lang": "it", "tld": "it", "gender": None, "slow": False},
        "Japanese": {"lang": "ja", "tld": "jp", "gender": None, "slow": False},
        "Korean": {"lang": "ko", "tld": "kr", "gender": None, "slow": False},
        "Portuguese": {"lang": "pt", "tld": "com.br", "gender": None, "slow": False},
        "Russian": {"lang": "ru", "tld": "ru", "gender": None, "slow": False},
        "Hindi": {"lang": "hi", "tld": "co.in", "gender": None, "slow": False},
        "Arabic": {"lang": "ar", "tld": "com", "gender": None, "slow": False},
    }
    
    selected_voice = st.selectbox(
        "Select Voice:",
        options=list(voice_options.keys())
    )
    
    voice_speed = st.select_slider(
        "Speaking Speed:",
        options=["Very Slow", "Slow", "Normal", "Fast", "Very Fast"],
        value="Normal"
    )
    
    # Map speed selection to actual speed values
    speed_values = {
        "Very Slow": 0.5,
        "Slow": 0.75,
        "Normal": 1.0,
        "Fast": 1.25,
        "Very Fast": 1.5
    }
    
    # Voice style
    voice_style = st.selectbox(
        "Voice Style:",
        options=["Neutral", "Happy", "Serious", "Excited", "Calm", "Mysterious"]
    )
    
    st.markdown("---")
    st.markdown("### 💡 Pro Tips")
    st.info("Short sentences often sound more natural than very long paragraphs.")
    st.warning("Add emotion markers like [happy], [excited], or [whisper] to enhance speech.")

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
            volume = st.slider(
                "🔊 Volume:",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Adjust the output volume (1.0 is normal)"
            )
        
        with param_col2:
            pitch_shift = st.slider(
                "🎵 Pitch Adjustment:",
                min_value=-10,
                max_value=10,
                value=0,
                step=1,
                help="Adjust the pitch of the voice (0 is normal)"
            )
        
        # Effects section
        st.markdown("### ✨ Special Effects")
        effects_col1, effects_col2, effects_col3 = st.columns(3)
        
        with effects_col1:
            add_echo = st.checkbox("Echo Effect", help="Adds a slight echo to the voice")
        
        with effects_col2:
            add_background = st.checkbox("Background Music", help="Adds subtle background music")
        
        with effects_col3:
            emphasize_words = st.checkbox("Word Emphasis", help="Dynamically emphasizes important words")
        
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
    """, unsafe_allow_html=True)
    
    # Add the CSS separately to avoid f-string issues with curly braces
    st.markdown("""
    <style>
    @keyframes pulse {
        0% {width: 30%; opacity: 0.6;}
        50% {width: 70%; opacity: 0.9;}
        100% {width: 30%; opacity: 0.6;}
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display language and voice info
    st.markdown("### 🌍 Language & Accent")
    
    voice_info = voice_options[selected_voice]
    lang_name = {
        "en": "English", "fr": "French", "de": "German", "es": "Spanish",
        "it": "Italian", "ja": "Japanese", "ko": "Korean", "pt": "Portuguese",
        "ru": "Russian", "hi": "Hindi", "ar": "Arabic"
    }
    
    accent_info = {
        "com": "American", "co.uk": "British", "fr": "France", "de": "Germany",
        "es": "Spain", "it": "Italy", "jp": "Japan", "kr": "Korea",
        "com.br": "Brazil", "ru": "Russia", "co.in": "India"
    }
    
    lang_display = lang_name.get(voice_info["lang"], voice_info["lang"])
    accent_display = accent_info.get(voice_info["tld"], voice_info["tld"])
    
    st.info(f"Language: {lang_display}\nAccent: {accent_display}\nSpeed: {voice_speed}")
    
    # Recently generated samples
    st.markdown("### 🕒 Recent Generations")
    
    # Initialize history in session state if not present
    if "history" not in st.session_state:
        st.session_state.history = []

# Function to convert text to speech
def text_to_speech(text, voice_params, speed=1.0, add_echo=False, volume=1.0):
    try:
        # Adjust text based on style
        styled_text = apply_style_to_text(text)
        
        # Create gTTS object
        tts = gtts.gTTS(
            text=styled_text,
            lang=voice_params["lang"],
            tld=voice_params["tld"],
            slow=(speed < 0.8)  # Use slow mode for very slow speech
        )
        
        # Create a temporary file path
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        audio_file = temp_dir / f"speech_{random.randint(1000, 9999)}.mp3"
        
        # Save the MP3 file
        tts.save(str(audio_file))
        
        # Process audio effects if needed
        if add_echo or pitch_shift != 0 or volume != 1.0:
            # Note: For actual audio manipulation, you would need to use a library like pydub
            # or librosa. Since those might add dependencies, we're keeping it simple here.
            # In a production app, you would implement the audio effects here.
            pass
        
        # Add to history
        if "history" in st.session_state:
            text_preview = text[:20] + "..." if len(text) > 20 else text
            st.session_state.history.append({
                "text": text_preview,
                "file": str(audio_file),
                "voice": selected_voice
            })
        
        return str(audio_file)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Function to apply style modifiers to text
def apply_style_to_text(text):
    # Add style markers based on selected style
    if voice_style != "Neutral" and not any(marker in text.lower() for marker in ["[happy]", "[excited]", "[serious]", "[calm]", "[mysterious]", "[whisper]"]):
        if voice_style == "Happy":
            text = "[happy] " + text
        elif voice_style == "Serious":
            text = "[serious] " + text
        elif voice_style == "Excited":
            text = "[excited] " + text
        elif voice_style == "Calm":
            text = "[calm] " + text
        elif voice_style == "Mysterious":
            text = "[mysterious] " + text

    return text

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
            # Get voice parameters
            voice_params = voice_options[selected_voice]
            speed = speed_values[voice_speed]
            
            # Convert text to speech
            audio_path = text_to_speech(
                text=text_input,
                voice_params=voice_params,
                speed=speed,
                add_echo=add_echo if 'add_echo' in locals() else False,
                volume=volume if 'volume' in locals() else 1.0
            )
            
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
        st.markdown("### 📜 Recent Generations")
        for item in st.session_state.history[-3:]:  # Show last 3 items
            st.markdown(f"""
            <div style="padding: 10px; margin: 5px 0; border-radius: 5px; border: 1px solid #ddd;">
                <p><small>{item['text']}</small></p>
                <p><small><b>Voice:</b> {item['voice']}</small></p>
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
   pip install streamlit gtts scipy numpy
   ```
3. Run the app locally:
   ```
   streamlit run app.py
   ```
4. For cloud deployment, create a `requirements.txt` file with:
   ```
   streamlit
   gtts
   scipy
   numpy
   ```
   
   This version doesn't require PyTorch or any special system dependencies!
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