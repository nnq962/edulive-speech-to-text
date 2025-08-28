import numpy as np
from typing import Tuple
from config import Config
from utils.logger_config import LOGGER

class VADProcessor:
    """Voice Activity Detection using simple RMS-based approach"""
    
    def __init__(self):
        self.silence_threshold = Config.SILENCE_THRESHOLD
        self.min_speech_frames = Config.MIN_SPEECH_FRAMES
        
        # Adaptive threshold (will adjust based on background noise)
        self.background_noise_level = 0
        self.noise_samples = []
        self.adaptation_frames = 0
        
        if Config.VERBOSE:
            LOGGER.info(f"VAD initialized with threshold: {self.silence_threshold}")
    
    def calculate_rms(self, audio_chunk: bytes) -> float:
        """Calculate RMS (Root Mean Square) for volume detection"""
        try:
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            if len(audio_array) == 0:
                return 0.0
            
            # Calculate RMS
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            return rms
        except Exception as e:
            if Config.VERBOSE:
                LOGGER.error(f"Error calculating RMS: {e}")
            return 0.0
    
    def adapt_threshold(self, rms_value: float, is_speech: bool):
        """Adaptively adjust threshold based on background noise"""
        if not is_speech and len(self.noise_samples) < 100:
            self.noise_samples.append(rms_value)
            
            if len(self.noise_samples) >= 50:
                # Update background noise level
                self.background_noise_level = np.mean(self.noise_samples[-50:])
                # Set threshold to be 3x background noise
                adaptive_threshold = max(self.background_noise_level * 3, Config.SILENCE_THRESHOLD)
                
                if abs(adaptive_threshold - self.silence_threshold) > 100:
                    self.silence_threshold = adaptive_threshold
                    if Config.VERBOSE:
                        LOGGER.info(f"Adapted VAD threshold to: {self.silence_threshold:.1f}")
    
    def is_speech(self, audio_chunk: bytes) -> Tuple[bool, float]:
        """
        Detect if audio chunk contains speech
        Returns: (is_speech, rms_value)
        """
        rms_value = self.calculate_rms(audio_chunk)
        
        # Simple threshold-based detection
        is_speech_detected = rms_value > self.silence_threshold
        
        # Adaptive threshold adjustment
        self.adapt_threshold(rms_value, is_speech_detected)
        
        if Config.VERBOSE and self.adaptation_frames % 100 == 0:
            LOGGER.debug(f"VAD - RMS: {rms_value:.1f}, Threshold: {self.silence_threshold:.1f}, Speech: {is_speech_detected}")
        
        self.adaptation_frames += 1
        
        return is_speech_detected, rms_value
    
    def reset(self):
        """Reset VAD state"""
        self.noise_samples.clear()
        self.adaptation_frames = 0
        self.background_noise_level = 0
        if Config.VERBOSE:
            LOGGER.info("VAD state reset")