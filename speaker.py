import os
import pickle
import torch
import whisper
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization

# Path to the pickle file
pickle_path = r"C:\Users\Danny\files\whisper_model.pkl"

# Function to load or create Whisper model
def get_whisper_model():
    if os.path.exists(pickle_path):
        print("Loading existing Whisper model from pickle file...")
        with open(pickle_path, 'rb') as f:
            model = pickle.load(f)
    else:
        print("Creating new Whisper large model...")
        model = whisper.load_model("large")
        # Save model for future use
        with open(pickle_path, 'wb') as f:
            pickle.dump(model, f)
    return model

# Function to transcribe audio with speaker diarization
def transcribe_with_speakers(audio_path,model=None):
    # Initialize pipeline with HuggingFace token

    if model is None:
        model = get_whisper_model()

    # Get the Whisper model and move to GPU if available
    
    if torch.cuda.is_available():
        model = model.to("cuda")
    

    # Transcribe the full audio
    result = model.transcribe(audio_path)
    
    # Process segments with speaker information

        
        # Find corresponding transcription
    for segment in result["segments"]:
        print(segment)
    
    return result["segments"]

# Example usage
if __name__ == "__main__":
    model = get_whisper_model()
    # Get filename from user
    filename = input("Enter the audio filename: ")
    while filename != "exit":
        filename = input("Enter the audio filename: ")
        audio_file = os.path.join(r"C:\Users\Danny\files", filename)

        # Check if file exists
        if not os.path.exists(audio_file):
            print(f"Error: File {audio_file} not found")
        else:
            transcription = transcribe_with_speakers(audio_file,model)
            for segment in transcription:
                print(f"Speaker {segment['id']}: {segment['text']}")