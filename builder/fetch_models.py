from concurrent.futures import ThreadPoolExecutor
import os
from faster_whisper import WhisperModel
import whisperx
from pyannote.audio import Pipeline
from huggingface_hub import hf_hub_download

whisper_model_names = ["tiny"]#TODO"large-v2", "large-v3"]
alignment_language_codes = ["en", "ru"]#"en", "fr", "de", "es", "it", "ja", "zh", "nl", "uk", "pt"]

def load_whisper_model(selected_model):
    print(f"Loading Whisper model: {selected_model}")
    for _attempt in range(5):
        try:
            loaded_model = WhisperModel(selected_model, device="cpu", compute_type="int8")
            print(f"Whisper model {selected_model} loaded successfully")
            return selected_model, loaded_model
        except (AttributeError, OSError) as e:
            print(f"Attempt to load {selected_model} failed: {str(e)}")
    
    print(f"Failed to load Whisper model {selected_model} after 5 attempts")
    return selected_model, None

def load_alignment_model(language_code):
    print(f"Loading alignment model for language: {language_code}")
    try:
        model = whisperx.load_align_model(language_code=language_code, device="cpu")
        print(f"Alignment model for {language_code} loaded successfully")
        return language_code, model
    except Exception as e:
        print(f"Failed to load alignment model for {language_code}: {str(e)}")
        return language_code, None

def load_diarization_model():
    print("Loading diarization model")
    try:
        model = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                         use_auth_token=os.environ.get("HUGGING_FACE_HUB_TOKEN"))
        print("Diarization model loaded successfully")
        return "diarization", model
    except Exception as e:
        print(f"Failed to load diarization model: {str(e)}")
        return "diarization", None

def download_huggingface_model(model_id):
    print(f"Downloading model from Hugging Face: {model_id}")
    try:
        hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
        print(f"Model {model_id} downloaded successfully")
        return model_id, True
    except Exception as e:
        print(f"Failed to download model {model_id}: {str(e)}")
        return model_id, False

whisper_models = {}
alignment_models = {}
diarization_model = None
huggingface_models = {}

# Load Whisper models
with ThreadPoolExecutor() as executor:
    for model_name, model in executor.map(load_whisper_model, whisper_model_names):
        if model is not None:
            whisper_models[model_name] = model

# Load alignment models
with ThreadPoolExecutor() as executor:
    for language_code, model in executor.map(load_alignment_model, alignment_language_codes):
        if model is not None:
            alignment_models[language_code] = model

# Load diarization model
_, diarization_model = load_diarization_model()

# Download additional Hugging Face models if needed
huggingface_model_ids = [
    "pyannote/speaker-diarization-3.1",
]

with ThreadPoolExecutor() as executor:
    for model_id, success in executor.map(download_huggingface_model, huggingface_model_ids):
        huggingface_models[model_id] = success

print("Model loading and caching completed.")
print(f"Loaded Whisper models: {list(whisper_models.keys())}")
print(f"Loaded alignment models: {list(alignment_models.keys())}")
print(f"Diarization model loaded: {diarization_model is not None}")
print(f"Downloaded Hugging Face models: {huggingface_models}")