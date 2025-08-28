import torch
from typing import Literal

class Config:
    # Server settings
    HOST = "0.0.0.0"
    PORT = 8808
    
    # Logging settings
    VERBOSE = False  # Set to False to disable verbose logging
    
    # Model settings
    PARTIAL_MODEL_SIZE = "small"  # For fast partial transcription
    FINAL_MODEL_SIZE = "medium"   # For accurate final transcription
    LANGUAGE = "vi"  # "vi", "en", or None for auto-detect
    
    # Device settings - auto detect GPU/CPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
    
    # Audio settings
    SAMPLE_RATE = 16000
    
    # VAD (Voice Activity Detection) settings
    SILENCE_THRESHOLD = 800     # RMS threshold for speech detection (adjust based on mic)
    MIN_SPEECH_FRAMES = 15      # Minimum frames to consider as speech
    SILENCE_COUNT_THRESHOLD = 80 # Frames of silence before ending speech (~4 seconds)
    
    # Transcription timing
    PARTIAL_INTERVAL = 1.5      # Seconds between partial transcriptions
    MIN_AUDIO_LENGTH = 0.3      # Minimum audio length (seconds) to transcribe
    TRANSCRIPTION_TIMEOUT = 30  # Max seconds to wait for transcription
    
    # Hallucination suppression
    SUPPRESS_PHRASES = [
        "hãy subscribe",
        "đăng ký kênh",
        "ghiền mì gõ",
        "like and subscribe",
        "subscribe cho kênh",
        "không bỏ lỡ",
        "video hấp dẫn"
    ]
    
    # WebSocket settings
    MAX_CONNECTIONS = 1  # Only allow 1 connection at a time
    PING_INTERVAL = 20   # Ping every 20 seconds
    PING_TIMEOUT = 10    # Wait 10 seconds for pong
    
    # Buffer settings
    MAX_AUDIO_BUFFER = 300  # Maximum audio buffer size (~10 seconds)
    
    @classmethod
    def log_config(cls):
        """Log current configuration"""
        return {
            "device": cls.DEVICE,
            "compute_type": cls.COMPUTE_TYPE,
            "partial_model": cls.PARTIAL_MODEL_SIZE,
            "final_model": cls.FINAL_MODEL_SIZE,
            "language": cls.LANGUAGE,
            "silence_threshold": cls.SILENCE_THRESHOLD,
            "verbose": cls.VERBOSE
        }