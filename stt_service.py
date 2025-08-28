import asyncio
import numpy as np
from typing import List, Optional, Tuple
from faster_whisper import WhisperModel
from datetime import datetime

from config import Config
from models import TranscriptionResponse
from utils.logger_config import LOGGER

class STTService:
    """Speech-to-Text service using faster-whisper"""
    
    def __init__(self):
        self.partial_model: Optional[WhisperModel] = None
        self.final_model: Optional[WhisperModel] = None
        self._model_lock = asyncio.Lock()
        
    async def initialize_models(self):
        """Initialize Whisper models"""
        try:
            if Config.VERBOSE:
                LOGGER.info("Initializing Whisper models...")
                LOGGER.info(f"Device: {Config.DEVICE}, Compute type: {Config.COMPUTE_TYPE}")
            
            # Initialize partial model (fast)
            self.partial_model = WhisperModel(
                Config.PARTIAL_MODEL_SIZE,
                device=Config.DEVICE,
                compute_type=Config.COMPUTE_TYPE,
                download_root=None,
                local_files_only=False
            )
            
            # Initialize final model (accurate)
            if Config.FINAL_MODEL_SIZE != Config.PARTIAL_MODEL_SIZE:
                self.final_model = WhisperModel(
                    Config.FINAL_MODEL_SIZE,
                    device=Config.DEVICE,
                    compute_type=Config.COMPUTE_TYPE,
                    download_root=None,
                    local_files_only=False
                )
            else:
                self.final_model = self.partial_model
            
            if Config.VERBOSE:
                LOGGER.info(f"Models initialized: {Config.PARTIAL_MODEL_SIZE} (partial), {Config.FINAL_MODEL_SIZE} (final)")
            
        except Exception as e:
            LOGGER.error(f"Failed to initialize models: {e}")
            raise
    
    def _suppress_hallucinations(self, text: str) -> Tuple[str, bool]:
        """
        Remove common hallucination patterns
        Returns: (cleaned_text, was_suppressed)
        """
        if not text or not text.strip():
            return text, False
        
        text_lower = text.lower()
        
        # Check for suppression phrases
        for phrase in Config.SUPPRESS_PHRASES:
            if phrase.lower() in text_lower:
                if Config.VERBOSE:
                    LOGGER.warning(f"Suppressed hallucination: '{text}'")
                return "", True
        
        # Check for repetitive patterns (same phrase repeated)
        words = text.split()
        if len(words) > 4:
            # Simple repetition detection
            for i in range(len(words) - 2):
                word_sequence = " ".join(words[i:i+3])
                remaining_text = " ".join(words[i+3:])
                if word_sequence in remaining_text:
                    if Config.VERBOSE:
                        LOGGER.warning(f"Suppressed repetitive text: '{text}'")
                    return "", True
        
        return text, False
    
    def _prepare_audio(self, audio_frames: List[bytes]) -> Optional[np.ndarray]:
        """Convert audio frames to numpy array for transcription"""
        try:
            if not audio_frames:
                return None
            
            # Combine all frames
            audio_data = b''.join(audio_frames)
            
            # Check minimum length
            min_bytes = int(Config.MIN_AUDIO_LENGTH * Config.SAMPLE_RATE * 2)  # 2 bytes per sample
            if len(audio_data) < min_bytes:
                return None
            
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert to float32 in range [-1, 1]
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            return audio_float
            
        except Exception as e:
            LOGGER.error(f"Error preparing audio: {e}")
            return None
    
    async def transcribe_partial(self, audio_frames: List[bytes]) -> TranscriptionResponse:
        """Perform fast partial transcription"""
        try:
            audio_data = self._prepare_audio(audio_frames)
            if audio_data is None:
                return TranscriptionResponse(
                    type="error",
                    timestamp=datetime.now().isoformat(),
                    error_message="Invalid audio data"
                )
            
            async with self._model_lock:
                # Use timeout for transcription
                transcription_task = asyncio.create_task(
                    asyncio.to_thread(self._transcribe_sync, self.partial_model, audio_data, True)
                )
                
                try:
                    text, confidence = await asyncio.wait_for(
                        transcription_task, 
                        timeout=Config.TRANSCRIPTION_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    transcription_task.cancel()
                    raise Exception("Partial transcription timeout")
            
            # Suppress hallucinations
            cleaned_text, was_suppressed = self._suppress_hallucinations(text)
            
            if was_suppressed or not cleaned_text.strip():
                return TranscriptionResponse(
                    type="partial",
                    text="",
                    timestamp=datetime.now().isoformat(),
                    confidence=0.0
                )
            
            if Config.VERBOSE:
                LOGGER.info(f"Partial transcription: '{cleaned_text}' (confidence: {confidence:.2f})")
            
            return TranscriptionResponse(
                type="partial",
                text=cleaned_text.strip(),
                timestamp=datetime.now().isoformat(),
                confidence=confidence
            )
            
        except Exception as e:
            LOGGER.error(f"Partial transcription error: {e}")
            return TranscriptionResponse(
                type="error",
                timestamp=datetime.now().isoformat(),
                error_message=f"Partial transcription failed: {str(e)}"
            )
    
    async def transcribe_final(self, audio_frames: List[bytes]) -> TranscriptionResponse:
        """Perform accurate final transcription"""
        try:
            audio_data = self._prepare_audio(audio_frames)
            if audio_data is None:
                return TranscriptionResponse(
                    type="error",
                    timestamp=datetime.now().isoformat(),
                    error_message="Invalid audio data"
                )
            
            async with self._model_lock:
                # Use timeout for transcription
                transcription_task = asyncio.create_task(
                    asyncio.to_thread(self._transcribe_sync, self.final_model, audio_data, False)
                )
                
                try:
                    text, confidence = await asyncio.wait_for(
                        transcription_task,
                        timeout=Config.TRANSCRIPTION_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    transcription_task.cancel()
                    raise Exception("Final transcription timeout")
            
            # Suppress hallucinations
            cleaned_text, was_suppressed = self._suppress_hallucinations(text)
            
            if was_suppressed:
                cleaned_text = ""
            
            if Config.VERBOSE:
                LOGGER.info(f"Final transcription: '{cleaned_text}' (confidence: {confidence:.2f})")
            
            return TranscriptionResponse(
                type="final",
                text=cleaned_text.strip(),
                timestamp=datetime.now().isoformat(),
                confidence=confidence
            )
            
        except Exception as e:
            LOGGER.error(f"Final transcription error: {e}")
            return TranscriptionResponse(
                type="error",
                timestamp=datetime.now().isoformat(),
                error_message=f"Final transcription failed: {str(e)}"
            )
    
    def _transcribe_sync(self, model: WhisperModel, audio_data: np.ndarray, is_partial: bool) -> Tuple[str, float]:
        """Synchronous transcription wrapper"""
        try:
            # Configure transcription parameters
            beam_size = 1 if is_partial else 5
            best_of = 1 if is_partial else 1
            temperature = 0.0
            
            segments, info = model.transcribe(
                audio_data,
                language=Config.LANGUAGE,
                beam_size=beam_size,
                best_of=best_of,
                temperature=temperature,
                vad_filter=False,  # We handle VAD ourselves
                vad_parameters=None
            )
            
            # Combine all segments
            full_text = ""
            total_confidence = 0.0
            segment_count = 0
            
            for segment in segments:
                full_text += segment.text + " "
                if hasattr(segment, 'avg_logprob'):
                    total_confidence += segment.avg_logprob
                    segment_count += 1
            
            # Calculate average confidence (convert from log prob to 0-1)
            if segment_count > 0:
                avg_confidence = total_confidence / segment_count
                confidence = max(0.0, min(1.0, (avg_confidence + 1.0)))  # Rough conversion
            else:
                confidence = 0.0
            
            return full_text.strip(), confidence
            
        except Exception as e:
            LOGGER.error(f"Sync transcription error: {e}")
            return "", 0.0
    
    async def cleanup(self):
        """Cleanup resources"""
        if Config.VERBOSE:
            LOGGER.info("Cleaning up STT service...")
        
        # Models will be garbage collected automatically
        self.partial_model = None
        self.final_model = None