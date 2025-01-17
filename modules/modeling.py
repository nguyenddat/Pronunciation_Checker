from typing import *

import torch
import torch.nn as nn
import pickle

from ..schemas import IasrModel
from ..models import neuralASR, whisperWrapper

def get_asr_model(language: str, use_whisper: bool = True) -> IasrModel.IASRModel:
    if use_whisper:
        return whisperWrapper.WhisperASRModel()
    
    if language not in {"de", "en"}:
        raise ValueError(f"Language {language} is not supported. Only 'de', 'en' are supported")

    model, decoder, utils = torch.hub.load(repo_or_dir = "snakers4/silero-models",
                                          model = "silero_stt",
                                          language = language,
                                          device = torch.device("cpu"))
    
    return neuralASR.NeuralASR(model, decoder)

def get_tts_model(language: str) -> nn.Module:
    if language not in {"de", "en"}:
        raise ValueError(f"Language {language} is not supported. Only 'de', 'en' are supported")
    

    