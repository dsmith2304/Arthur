import sounddevice as sd
import numpy as np
import wave
import os
from pathlib import Path
import pyttsx3
import speech_recognition as sr
import whisper
from pydub import AudioSegment
import pickle
import time
import speaker
def text_to_speech(text):
    
    engine = pyttsx3.init()
    engine.runAndWait()
    engine.setProperty('rate', 250)
    engine.say(text)
    engine.runAndWait()

def record_until_silence(silence_duration=3, threshold=0.0025, samplerate=8000, channels=1,file:str="data.wav"):
    """
    Record audio until 3 seconds of silence is detected.
    
    Parameters:
    - silence_duration: The duration of silence (in seconds) to wait before stopping.
    - threshold: The threshold below which audio is considered silence.
    - samplerate: The sampling rate of the audio recording.
    - channels: The number of audio channels (1 for mono, 2 for stereo).
    """
    # Duration of each chunk of audio to record in seconds
    chunk_duration = 1
    chunk_size = int(samplerate * chunk_duration)

    # Initialize an empty list to hold audio data
    audio_data = []


    # Variable to track consecutive silent chunks
    silent_chunks = 0
    fs = 8000
        # Create a flag to track if sound is detected
    sound_detected = False
    initial_audio = []
    flag=False

    def callback(indata, frames, time, status):
        if not flag:
            nonlocal sound_detected, initial_audio
            volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))
            if volume_norm > threshold:
                print(f"Volume: {volume_norm}")
                sound_detected = True
                initial_audio.append(indata.copy())
        else:
            audio_data.append(indata.copy())
    # Start the audio stream

    with sd.InputStream(callback=callback, channels=channels, samplerate=samplerate):
        print("Recording... Press Ctrl+C to stop.")
        while not sound_detected:
            time.sleep(0.0005)

        try:
            flag=True
            threshold= threshold/4
            while True:

                time.sleep(0.0005)
                # Convert audio_data to a numpy array and check for silence
                if len(audio_data) > 0:
                    chunk = np.concatenate(audio_data[-1], axis=0)
                    

                    # Calculate the average amplitude of the chunk
                    avg_amplitude = np.mean(np.abs(chunk))

                    # Check if the average amplitude is below the silence threshold
                    if avg_amplitude < threshold:
                        silent_chunks += 1
                    else:
                        print(f"Amplitude: {avg_amplitude}")
                        print(len(audio_data))
                        silent_chunks = 0  # Reset counter if sound is detected

                    # If 3 seconds of silence (3 chunks) have been detected, stop recording
                    if silent_chunks >= silence_duration:
                        print("Detected 3 seconds of silence. Stopping...")
                        break
        
        except KeyboardInterrupt:
            print("\nRecording stopped manually.")

    home = str(Path.home())
    file_path = os.path.join(home, 'files')
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_path = os.path.join(file_path, file)
    wavefile = wave.open(file_path, 'wb')
    wavefile.setnchannels(1)
    wavefile.setsampwidth(2)
    wavefile.setframerate(fs)
    wavefile.writeframes((np.array(np.concatenate(audio_data, axis=0)) * 32767).astype(np.int16).tobytes())
    wavefile.close()

def recording_to_text(file:str="data.wav",model:whisper.model=None):
    home = str(Path.home())
    file_path = os.path.join(home, 'files')
    
    # Load Whisper model and transcribe with fp32
    model_pickle_path = os.path.join(file_path, 'whisper_model.pkl')
    
    if model is None:
        if os.path.exists(model_pickle_path):
            with open(model_pickle_path, 'rb') as f:
                model = pickle.load(f)
        else:
            model = whisper.load_model("large")
            with open(model_pickle_path, 'wb') as f:
                pickle.dump(model, f)
    model.to("cuda")
    result = model.transcribe(os.path.join(file_path, file))

    return result["text"]

def record_audio(file:str="data.wav"):

    # listen to mic. If a sound is picked up start recording
    # if sound stops ensure it has stopped for an entire 3 seconds before stopping recording
    # save recording to file
    fs = 8000
    threshold = 0.025 # Adjust this value based on your needs
    SILENCE_DURATION = 10  # seconds
    flag:bool = False
    # Create a flag to track if sound is detected
    sound_detected = False
    initial_audio = []

    def callback(indata, frames, time, status):
        nonlocal sound_detected, initial_audio
        volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))
        if volume_norm > threshold:
            print(f"Volume: {volume_norm}")
            sound_detected = True
            initial_audio.append(indata.copy())

    # Wait for sound to start
    print("Waiting for speech to begin...")
    with sd.InputStream(callback=callback, channels=1, samplerate=fs,):
        while not sound_detected:
            time.sleep(0.0005)
            
    print("Sound detected, starting recording...")

    # Convert initial audio to numpy array and pass to record_until_silence
    initial_audio_array = np.concatenate(initial_audio, axis=0) if initial_audio else None
    recording = record_until_silence(existing_audio=initial_audio_array, 
                                   silence_duration=SILENCE_DURATION*1000, 
                                   threshold=threshold/4, 
                                   samplerate=fs,file=file)

    # Save recording to file
    home = str(Path.home())
    file_path = os.path.join(home, 'files')
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_path = os.path.join(file_path, file)
    wavefile = wave.open(file_path, 'wb')
    wavefile.setnchannels(1)
    wavefile.setsampwidth(2)
    wavefile.setframerate(fs)
    wavefile.writeframes((np.array(recording) * 32767).astype(np.int16).tobytes())
    wavefile.close()

def main():
    model = speaker.get_whisper_model()
    threshold = 0.025 # Adjust this value based on your needs
    SILENCE_DURATION = 10  # seconds
    fs = 8000 
    threshold = 0.025 # Adjust this value based on your needs
    record_until_silence(
                                   silence_duration=SILENCE_DURATION*1000, 
                                   threshold=threshold, 
                                   samplerate=fs,file="1.wav")
    value_spoken=speaker.transcribe_with_speakers(r"C:\Users\Danny\files\1.wav",model)

    for segment in value_spoken:
        print(segment['text'])




if __name__ == "__main__":
    main()