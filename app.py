import streamlit as st
import os
import json
import numpy as np
import librosa
import soundfile as sf
import warnings
from groq import Groq
import tempfile
from pyannote.audio import Pipeline
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(
    page_title="AI News Company Pipeline",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("AI News Company Pipeline")
st.markdown("""
This application converts speech to text with speaker diarization and generates news content.
Upload an audio or video file to get started.
""")

# Initialize session state variables if they don't exist
if 'pipeline_completed' not in st.session_state:
    st.session_state.pipeline_completed = False
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'generated_content' not in st.session_state:
    st.session_state.generated_content = None
if 'speaker_names' not in st.session_state:
    st.session_state.speaker_names = None
if 'file_paths' not in st.session_state:
    st.session_state.file_paths = {}

# Create directories for storing temporary files
temp_dir = tempfile.mkdtemp()
input_dir = os.path.join(temp_dir, "input")
output_dir = os.path.join(temp_dir, "output")
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Sidebar for API key and configuration
st.sidebar.title("Configuration")
groq_api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")
use_whisper_api = st.sidebar.checkbox("Use Whisper API via Groq", value=True)

# Function to get Groq client
def get_groq_client():
    if groq_api_key:
        return Groq(api_key=groq_api_key)
    else:
        st.sidebar.error("Please enter a valid Groq API Key")
        return None

# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    """Save uploaded file to the input directory and return the path"""
    file_path = os.path.join(input_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Function to test Groq API connection
def test_groq_connection():
    try:
        groq_client = get_groq_client()
        if groq_client:
            test_response = groq_client.chat.completions.create(
                model="llama3-8b-8192",  # Using smaller model for quick test
                messages=[
                    {"role": "user", "content": "Hello, can you hear me?"}
                ],
                max_tokens=10
            )
            st.sidebar.success("Groq API connection successful!")
            return True
        return False
    except Exception as e:
        st.sidebar.error(f"Could not connect to Groq API: {str(e)}")
        return False

# Function to transcode video to audio
def extract_audio_from_video(video_path, audio_path):
    """Extract audio from video using ffmpeg"""
    import subprocess
    try:
        # Use ffmpeg to extract audio
        command = [
            "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", 
            "-ar", "16000", "-ac", "1", audio_path, "-y", "-loglevel", "error"
        ]
        subprocess.run(command, check=True)
        
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            st.error("Failed to extract audio from video")
            return False
        return True
    except Exception as e:
        st.error(f"Error extracting audio: {str(e)}")
        return False

# Speaker Diarization Function
def transcribe_with_speaker_diarization(audio_path, output_path, use_whisper_api=True):
    """Transcribe audio file with speaker diarization"""
    try:
        with st.spinner("Loading speaker diarization model..."):
            # Initialize the speaker diarization pipeline
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization@2.1",
                use_auth_token=True
            )

        with st.spinner("Running speaker diarization..."):
            # Run speaker diarization on the audio file
            diarization_result = diarization_pipeline(audio_path)

            # Load audio file
            audio, sr = librosa.load(audio_path, sr=16000)

            # Create a dictionary to store segments for each speaker
            speaker_segments = {}

            # Process diarization result
            for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                # Extract audio segment for the current speaker
                start_sample = int(turn.start * sr)
                end_sample = int(turn.end * sr)

                # Skip invalid segments
                if start_sample >= end_sample or start_sample >= len(audio) or end_sample > len(audio):
                    continue

                segment = audio[start_sample:end_sample]

                # Append segment to the speaker's dictionary
                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                speaker_segments[speaker].append({
                    "start": turn.start,
                    "end": turn.end,
                    "audio": segment
                })

        # Process each speaker's segments
        temp_segment_files = {}
        processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="tamil", task="transcribe")

        progress_bar = st.progress(0)
        total_speakers = len(speaker_segments)
        
        for i, (speaker, segments) in enumerate(speaker_segments.items()):
            speaker_transcripts = []
            st.text(f"Processing speaker {speaker}...")

            # Process segments in batches
            batch_size = 5  # Adjust as needed
            for j in range(0, len(segments), batch_size):
                batch = segments[j:j+batch_size]

                # Concatenate batch segments for efficiency
                batch_audio = np.concatenate([seg["audio"] for seg in batch])
                batch_file = os.path.join(input_dir, f"temp_speaker_{speaker}_batch_{j}.wav")
                sf.write(batch_file, batch_audio, sr)
                temp_segment_files[batch_file] = True

                # Transcribe batch
                input_features = processor(batch_audio, sampling_rate=sr, return_tensors="pt").input_features

                # Generate token ids
                with torch.no_grad():
                    predicted_ids = model.generate(input_features, max_length=256)

                # Decode the token ids to text
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True, normalize=True)[0]

                # Add the transcription to this speaker's list
                speaker_transcripts.append({
                    "start": batch[0]["start"],
                    "end": batch[-1]["end"],
                    "text": transcription
                })
            
            # Store this speaker's transcripts
            speaker_segments[speaker] = speaker_transcripts
            progress_value = (i + 1) / total_speakers
            progress_bar.progress(progress_value)

        # Write transcripts to file
        with open(output_path, 'w', encoding='utf-8') as f:
            for speaker, transcripts in speaker_segments.items():
                f.write(f"Speaker {speaker}:\n")
                for transcript in transcripts:
                    f.write(f"[{transcript['start']:.2f} - {transcript['end']:.2f}] {transcript['text']}\n")
                f.write("\n")

        # Clean up temporary files
        for temp_file in temp_segment_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        progress_bar.empty()
        st.success(f"Transcript completed and saved")
        return speaker_segments

    except Exception as e:
        st.error(f"Error in speaker diarization: {str(e)}")
        # Fall back to basic transcription without diarization
        return basic_transcription(audio_path, output_path)

# Basic transcription function as fallback
def basic_transcription(audio_path, output_path):
    """Perform basic transcription without speaker diarization as a fallback"""
    st.warning("Falling back to basic transcription without speaker diarization...")

    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)

        # Process in chunks to avoid memory issues
        chunk_length_s = 30  # Process 30 seconds at a time
        chunk_length = chunk_length_s * sr

        chunks = [audio[i:i+chunk_length] for i in range(0, len(audio), chunk_length)]

        full_transcript = []

        # Initialize Whisper model
        processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="tamil", task="transcribe")

        progress_bar = st.progress(0)
        
        for i, chunk in enumerate(chunks):
            # Save chunk to temp file
            temp_chunk_file = os.path.join(input_dir, f"temp_chunk_{i}.wav")
            sf.write(temp_chunk_file, chunk, sr)

            # Process with Whisper
            input_features = processor(chunk, sampling_rate=sr, return_tensors="pt").input_features

            # Generate token ids
            with torch.no_grad():
                predicted_ids = model.generate(input_features, max_length=256)

            # Decode the token ids to text
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True, normalize=True)[0]

            # Clean up temp file
            if os.path.exists(temp_chunk_file):
                os.remove(temp_chunk_file)

            start_time = i * chunk_length_s
            end_time = min((i + 1) * chunk_length_s, len(audio) / sr)

            full_transcript.append({
                "start": start_time,
                "end": end_time,
                "text": transcription
            })
            
            # Update progress
            progress_value = (i + 1) / len(chunks)
            progress_bar.progress(progress_value)

        # Create a simple speaker transcript with all text assigned to one speaker
        speaker_transcripts = {"UNKNOWN": full_transcript}

        # Write transcript to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Speaker UNKNOWN:\n")
            for transcript in full_transcript:
                f.write(f"[{transcript['start']:.2f} - {transcript['end']:.2f}] {transcript['text']}\n")

        progress_bar.empty()
        st.success(f"Basic transcript saved")
        return speaker_transcripts

    except Exception as e:
        st.error(f"Error in basic transcription: {str(e)}")
        # Create a minimal transcript to allow the pipeline to continue
        speaker_transcripts = {"UNKNOWN": [{"start": 0, "end": 1, "text": "Transcription failed."}]}

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Speaker UNKNOWN:\n")
            f.write("[0.00 - 1.00] Transcription failed. Please check the audio file.")

        return speaker_transcripts

# Content Generation Function
def generate_news_content(transcript, speaker_names=None):
    """Generate news content based on the transcript using Groq API"""
    try:
        st.subheader("Content Generation")
        
        # Initialize Groq client
        groq_client = get_groq_client()
        if not groq_client:
            st.error("Content generation requires a valid Groq API key.")
            return None, speaker_names

        # If speaker names are not provided, use a form to collect them
        if not speaker_names:
            speaker_names = {}
            with st.expander("Provide speaker names (optional)"):
                st.write("Enter names for each speaker or leave blank to use default:")
                for speaker in transcript.keys():
                    speaker_names[speaker] = st.text_input(f"Name for Speaker {speaker}", value="", key=f"speaker_{speaker}")
                    if not speaker_names[speaker]:
                        speaker_names[speaker] = f"Speaker {speaker}"
        
        # Format the transcript for the model
        formatted_transcript = ""
        for speaker, segments in transcript.items():
            speaker_name = speaker_names.get(speaker, f"Speaker {speaker}")
            for segment in segments:
                formatted_transcript += f"{speaker_name}: {segment['text']}\n"

        # Truncate transcript if it's too long
        if len(formatted_transcript) > 5000:
            st.warning("Transcript is too long. Truncating to 5000 characters...")
            formatted_transcript = formatted_transcript[:5000] + "\n[Transcript truncated due to length]"

        generated_content = {}

        # Generate newspaper article
        with st.spinner("Generating newspaper article..."):
            article_prompt = (
                "You are a professional journalist. Write a formal newspaper article based on the following interview transcript. "
                "The article should be in code-mixed Tamil (mix of Tamil and English), formatted properly with a headline, "
                "introduction, body, and conclusion. Make it informative and engaging.\n\n"
                f"TRANSCRIPT:\n{formatted_transcript}\n\n"
                "Write the newspaper article:"
            )

            try:
                # Using Groq API to access LLaMA 3
                response = groq_client.chat.completions.create(
                    model="llama3-70b-8192",  # Using LLaMA 3 70B model
                    messages=[
                        {"role": "system", "content": "You are a skilled journalist who writes articles in code-mixed Tamil and English."},
                        {"role": "user", "content": article_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1500,
                    top_p=0.9
                )
                newspaper_article = response.choices[0].message.content
                generated_content["newspaper_article"] = newspaper_article
            except Exception as e:
                st.error(f"Error generating newspaper article: {str(e)}")
                generated_content["newspaper_article"] = "Failed to generate newspaper article."

        # Generate social media bite
        with st.spinner("Generating social media post..."):
            social_prompt = (
                "You are a social media content creator for a news channel. Write a short, engaging social media post "
                "(around 280 characters) based on the following interview transcript. The post should be in code-mixed Tamil "
                "(mix of Tamil and English) and should capture the essence of the interview.\n\n"
                f"TRANSCRIPT:\n{formatted_transcript}\n\n"
                "Write the social media post:"
            )

            try:
                response = groq_client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {"role": "system", "content": "You are a social media content creator who writes in code-mixed Tamil and English."},
                        {"role": "user", "content": social_prompt}
                    ],
                    temperature=0.8,
                    max_tokens=300,
                    top_p=0.9
                )
                social_media_bite = response.choices[0].message.content
                generated_content["social_media_bite"] = social_media_bite
            except Exception as e:
                st.error(f"Error generating social media post: {str(e)}")
                generated_content["social_media_bite"] = "Failed to generate social media post."

        # Generate news reader script
        with st.spinner("Generating news reader script..."):
            script_prompt = (
                "You are a script writer for a news channel. Write a script for news readers based on the following "
                "interview transcript. The script should be in code-mixed Tamil (mix of Tamil and English) and should include "
                "prompts for two news readers (Anchor 1 and Anchor 2) to read alternately.\n\n"
                f"TRANSCRIPT:\n{formatted_transcript}\n\n"
                "Write the news reader script:"
            )

            try:
                response = groq_client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {"role": "system", "content": "You are a script writer who writes in code-mixed Tamil and English."},
                        {"role": "user", "content": script_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1500,
                    top_p=0.9
                )
                news_reader_script = response.choices[0].message.content
                generated_content["news_reader_script"] = news_reader_script
            except Exception as e:
                st.error(f"Error generating news reader script: {str(e)}")
                generated_content["news_reader_script"] = "Failed to generate news reader script."

        st.success("Content generation completed successfully.")
        return generated_content, speaker_names

    except Exception as e:
        st.error(f"Error in content generation: {str(e)}")
        # Return minimal content to allow the pipeline to continue
        return {
            "newspaper_article": "Content generation failed. Please check the model and transcript.",
            "social_media_bite": "Content generation failed.",
            "news_reader_script": "Content generation failed."
        }, speaker_names or {"UNKNOWN": "Unknown Speaker"}

# Function to save generated content
def save_generated_content(generated_content, output_path):
    """Save generated content to files"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        file_paths = {}

        # Save newspaper article
        article_path = f"{output_path}_article.txt"
        with open(article_path, 'w', encoding='utf-8') as f:
            f.write(generated_content["newspaper_article"])
        file_paths["article"] = article_path

        # Save social media bite
        social_path = f"{output_path}_social.txt"
        with open(social_path, 'w', encoding='utf-8') as f:
            f.write(generated_content["social_media_bite"])
        file_paths["social"] = social_path

        # Save news reader script
        script_path = f"{output_path}_script.txt"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(generated_content["news_reader_script"])
        file_paths["script"] = script_path

        st.success("Generated content saved successfully.")
        return file_paths

    except Exception as e:
        st.error(f"Error saving generated content: {str(e)}")
        return {}

# Main app function
def main():
    # Check if API key test button is clicked
    if st.sidebar.button("Test Groq API Connection"):
        test_groq_connection()

    # File upload section
    st.subheader("Upload Audio/Video File")
    uploaded_file = st.file_uploader("Choose an audio or video file", type=["mp3", "wav", "mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        # Save uploaded file
        input_path = save_uploaded_file(uploaded_file)
        st.success(f"File {uploaded_file.name} uploaded successfully!")

        # Add a run button to start processing
        run_pipeline = st.button("Run AI News Pipeline")

        if run_pipeline:
            with st.spinner("Processing input file..."):
                # Process input file
                transcript_path = os.path.join(output_dir, "transcript.txt")
                output_base_path = os.path.join(output_dir, "news_content")

                # Process audio or video
                if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    st.info("Processing video input...")
                    # Extract audio from video
                    audio_path = os.path.join(input_dir, os.path.splitext(uploaded_file.name)[0] + '.wav')
                    if extract_audio_from_video(input_path, audio_path):
                        speaker_transcripts = transcribe_with_speaker_diarization(audio_path, transcript_path, use_whisper_api)
                    else:
                        st.error("Failed to extract audio from video.")
                        return
                else:
                    st.info("Processing audio input...")
                    speaker_transcripts = transcribe_with_speaker_diarization(input_path, transcript_path, use_whisper_api)

                # Check if transcription succeeded
                if not speaker_transcripts or all(len(segments) == 0 for speaker, segments in speaker_transcripts.items()):
                    st.error("Transcription failed or produced empty results. Please check your audio/video file.")
                    return

                # Store transcript in session state
                st.session_state.transcript = speaker_transcripts

                # Generate content
                generated_content, speaker_names = generate_news_content(speaker_transcripts)
                
                # Store generated content and speaker names in session state
                st.session_state.generated_content = generated_content
                st.session_state.speaker_names = speaker_names

                # Save generated content
                file_paths = save_generated_content(generated_content, output_base_path)
                
                # Save speaker names for future reference
                speaker_names_path = f"{output_base_path}_speakers.json"
                with open(speaker_names_path, 'w', encoding='utf-8') as f:
                    json.dump(speaker_names, f, ensure_ascii=False, indent=2)
                file_paths["speakers"] = speaker_names_path
                
                # Store file paths in session state
                st.session_state.file_paths = file_paths
                
                # Mark pipeline as completed
                st.session_state.pipeline_completed = True

    # Display results if pipeline has completed
    if st.session_state.pipeline_completed:
        st.header("Results")
        
        # Display transcript
        if st.session_state.transcript:
            with st.expander("Transcript", expanded=False):
                for speaker, segments in st.session_state.transcript.items():
                    speaker_name = st.session_state.speaker_names.get(speaker, f"Speaker {speaker}")
                    st.subheader(speaker_name)
                    for segment in segments:
                        st.write(f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}")
        
        # Display generated content
        if st.session_state.generated_content:
            tabs = st.tabs(["Newspaper Article", "Social Media Post", "News Reader Script"])
            
            with tabs[0]:
                st.markdown(st.session_state.generated_content["newspaper_article"])
                if "article" in st.session_state.file_paths:
                    with open(st.session_state.file_paths["article"], "r", encoding="utf-8") as f:
                        article_content = f.read()
                    st.download_button(
                        label="Download Article",
                        data=article_content,
                        file_name="newspaper_article.txt",
                        mime="text/plain"
                    )
            
            with tabs[1]:
                st.markdown(st.session_state.generated_content["social_media_bite"])
                if "social" in st.session_state.file_paths:
                    with open(st.session_state.file_paths["social"], "r", encoding="utf-8") as f:
                        social_content = f.read()
                    st.download_button(
                        label="Download Social Media Post",
                        data=social_content,
                        file_name="social_media_post.txt",
                        mime="text/plain"
                    )
            
            with tabs[2]:
                st.markdown(st.session_state.generated_content["news_reader_script"])
                if "script" in st.session_state.file_paths:
                    with open(st.session_state.file_paths["script"], "r", encoding="utf-8") as f:
                        script_content = f.read()
                    st.download_button(
                        label="Download News Reader Script",
                        data=script_content,
                        file_name="news_reader_script.txt",
                        mime="text/plain"
                    )

        # Download transcript button
        if "transcript" in st.session_state:
            transcript_path = os.path.join(output_dir, "transcript.txt")
            if os.path.exists(transcript_path):
                with open(transcript_path, "r", encoding="utf-8") as f:
                    transcript_content = f.read()
                st.sidebar.download_button(
                    label="Download Full Transcript",
                    data=transcript_content,
                    file_name="transcript.txt",
                    mime="text/plain"
                )

# Run the main app function
if __name__ == "__main__":
    main()