"""
This file contains the Predictor class, which is used to run predictions on the
WhisperX model with diarization support.
"""

from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import whisperx
from pyannote.audio import Pipeline

class Predictor:
    def __init__(self):
        self.models = {}
        self.align_models = {}
        self.diarization_pipeline = None

    def setup(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        model_names = ["tiny"]  # Можно добавить другие модели, если нужно
        
        for model_name in model_names:
            self.models[model_name] = whisperx.load_model(model_name, device, compute_type=compute_type,
                asr_options={
                    "max_new_tokens": 128,
                    "clip_timestamps": True,
                    "hallucination_silence_threshold": 2.0,
                    "hotwords": []
                })
        
        # Инициализация модели диаризации pyannote
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="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"
        )
        # Отправляем модель диаризации на GPU, если доступно
        if device == "cuda":
            self.diarization_pipeline.to(torch.device("cuda"))

    def predict(
        self,
        audio,
        model_name="tiny",
        language=None,
        diarize=False,
        min_speakers=None,
        max_speakers=None,
        num_speakers=None,
        batch_size=16,
        **kwargs
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = self.models[model_name]
        
        # 1. Транскрибирование
        audio_data = whisperx.load_audio(audio)
        result = model.transcribe(audio_data, batch_size=batch_size, **kwargs)
        
        # 2. Выравнивание
        if language is None:
            language = result["language"]
        
        if language not in self.align_models:
            self.align_models[language], metadata = whisperx.load_align_model(language_code=language, device=device)
        
        align_model = self.align_models[language]
        result = whisperx.align(result["segments"], align_model, metadata, audio_data, device, return_char_alignments=False)
        
        # 3. Диаризация (если требуется)
        if diarize:
            diarize_kwargs = {}
            if num_speakers is not None:
                diarize_kwargs['num_speakers'] = num_speakers
            elif min_speakers is not None or max_speakers is not None:
                if min_speakers is not None:
                    diarize_kwargs['min_speakers'] = min_speakers
                if max_speakers is not None:
                    diarize_kwargs['max_speakers'] = max_speakers
            
            diarization = self.diarization_pipeline(audio, **diarize_kwargs)
            result = whisperx.assign_word_speakers(diarization, result)
        
        return self.format_result(result)

    def format_result(self, result):
        formatted_result = {
            "segments": [],
            "language": result.get("language")
        }

        for segment in result["segments"]:
            formatted_segment = {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "speaker": segment.get("speaker"),
                "words": [
                    {
                        "word": word["word"],
                        "start": word["start"],
                        "end": word["end"],
                        "speaker": word.get("speaker")
                    }
                    for word in segment.get("words", [])
                ]
            }
            formatted_result["segments"].append(formatted_segment)

        return formatted_result
    
MODEL = Predictor()
MODEL.setup()