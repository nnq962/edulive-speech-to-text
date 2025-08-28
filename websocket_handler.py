import asyncio
import json
import time
from typing import Optional, List
from collections import deque
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect

from config import Config
from models import TranscriptionResponse, ConnectionStatus
from stt_service import STTService
from vad_processor import VADProcessor
from utils.logger_config import LOGGER

class WebSocketHandler:
    """Handle WebSocket connections and audio processing"""
    
    def __init__(self, stt_service: STTService):
        self.stt_service = stt_service
        self.current_connection: Optional[WebSocket] = None
        self.connection_status: Optional[ConnectionStatus] = None
        
        # Audio processing state
        self.audio_buffer = deque(maxlen=Config.MAX_AUDIO_BUFFER)
        self.vad_processor = VADProcessor()
        
        # Speech state
        self.is_speaking = False
        self.speech_frames: List[bytes] = []
        self.silence_count = 0
        self.last_partial_time = 0.0
        
        if Config.VERBOSE:
            LOGGER.info("WebSocket handler initialized")
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Handle new WebSocket connection"""
        # Disconnect existing connection if any
        if self.current_connection:
            try:
                await self._send_message(TranscriptionResponse(
                    type="status",
                    text="Connection replaced by new client",
                    timestamp=datetime.now().isoformat()
                ))
                await self.current_connection.close()
                if Config.VERBOSE:
                    LOGGER.info("Closed existing connection")
            except Exception as e:
                if Config.VERBOSE:
                    LOGGER.warning(f"Error closing existing connection: {e}")
        
        # Accept new connection
        await websocket.accept()
        self.current_connection = websocket
        self.connection_status = ConnectionStatus(
            client_id=client_id,
            connected_at=datetime.now()
        )
        
        # Reset state
        self._reset_audio_state()
        self.vad_processor.reset()
        
        if Config.VERBOSE:
            LOGGER.info(f"New client connected: {client_id}")
        
        # Send welcome message
        await self._send_message(TranscriptionResponse(
            type="status",
            text="Connected to STT service",
            timestamp=datetime.now().isoformat()
        ))
    
    async def disconnect(self):
        """Handle WebSocket disconnection"""
        if self.current_connection:
            self.current_connection = None
            self.connection_status = None
            self._reset_audio_state()
            
            if Config.VERBOSE:
                LOGGER.info("Client disconnected")
    
    async def handle_message(self, websocket: WebSocket, message):
        """Handle incoming WebSocket messages"""
        try:
            if isinstance(message, bytes):
                # Audio data received
                await self._process_audio_chunk(message)
            else:
                # Control message
                try:
                    data = json.loads(message)
                    await self._handle_control_message(data)
                except json.JSONDecodeError:
                    if Config.VERBOSE:
                        LOGGER.warning("Received invalid JSON message")
        
        except WebSocketDisconnect:
            await self.disconnect()
        except Exception as e:
            LOGGER.error(f"Error handling message: {e}")
            await self._send_error("Message processing error")
    
    async def _handle_control_message(self, data: dict):
        """Handle control messages from client"""
        action = data.get("action")
        
        if action == "start":
            self._reset_audio_state()
            await self._send_message(TranscriptionResponse(
                type="status",
                text="Ready to receive audio",
                timestamp=datetime.now().isoformat()
            ))
            if Config.VERBOSE:
                LOGGER.info("Audio recording started")
        
        elif action == "stop":
            if self.is_speaking and self.speech_frames:
                # Send final transcription for remaining audio
                await self._process_final_transcription()
            
            self._reset_audio_state()
            await self._send_message(TranscriptionResponse(
                type="status",
                text="Audio recording stopped",
                timestamp=datetime.now().isoformat()
            ))
            if Config.VERBOSE:
                LOGGER.info("Audio recording stopped")
        
        else:
            if Config.VERBOSE:
                LOGGER.warning(f"Unknown control action: {action}")
    
    async def _process_audio_chunk(self, audio_data: bytes):
        """Process incoming audio chunk"""
        if not self.current_connection:
            return
        
        # Add to buffer
        self.audio_buffer.append(audio_data)
        
        # Voice activity detection
        has_speech, rms_value = self.vad_processor.is_speech(audio_data)
        
        if has_speech:
            await self._handle_speech_detected(audio_data)
        else:
            await self._handle_silence_detected(audio_data)
        
        # Update connection status
        if self.connection_status:
            self.connection_status.frames_processed += 1
            self.connection_status.is_speaking = self.is_speaking
    
    async def _handle_speech_detected(self, audio_data: bytes):
        """Handle speech detection"""
        if not self.is_speaking:
            # Speech started
            self.is_speaking = True
            self.speech_frames.clear()
            self.silence_count = 0
            self.last_partial_time = time.time()
            
            await self._send_message(TranscriptionResponse(
                type="speech_start",
                timestamp=datetime.now().isoformat()
            ))
            
            if Config.VERBOSE:
                LOGGER.info("Speech started")
        
        # Add frame to speech buffer
        self.speech_frames.append(audio_data)
        self.silence_count = 0
        
        # Check for partial transcription
        current_time = time.time()
        if (current_time - self.last_partial_time >= Config.PARTIAL_INTERVAL and 
            len(self.speech_frames) > Config.MIN_SPEECH_FRAMES):
            
            await self._process_partial_transcription()
            self.last_partial_time = current_time
    
    async def _handle_silence_detected(self, audio_data: bytes):
        """Handle silence detection"""
        if self.is_speaking:
            self.silence_count += 1
            self.speech_frames.append(audio_data)  # Keep some silence for context
            
            # Check if speech has ended
            if self.silence_count >= Config.SILENCE_COUNT_THRESHOLD:
                await self._process_final_transcription()
                await self._end_speech_session()
    
    async def _process_partial_transcription(self):
        """Process partial transcription"""
        if not self.speech_frames:
            return
        
        try:
            # Use recent frames for partial transcription
            recent_frames = self.speech_frames[-60:]  # Last ~3 seconds
            response = await self.stt_service.transcribe_partial(recent_frames)
            
            if response.text:  # Only send if there's actual text
                await self._send_message(response)
        
        except Exception as e:
            LOGGER.error(f"Partial transcription error: {e}")
    
    async def _process_final_transcription(self):
        """Process final transcription"""
        if not self.speech_frames:
            return
        
        try:
            response = await self.stt_service.transcribe_final(self.speech_frames)
            await self._send_message(response)
        
        except Exception as e:
            LOGGER.error(f"Final transcription error: {e}")
    
    async def _end_speech_session(self):
        """End current speech session"""
        self.is_speaking = False
        self.speech_frames.clear()
        self.silence_count = 0
        
        await self._send_message(TranscriptionResponse(
            type="speech_end",
            timestamp=datetime.now().isoformat()
        ))
        
        if Config.VERBOSE:
            LOGGER.info("Speech ended")
    
    async def _send_message(self, response: TranscriptionResponse):
        """Send message to client"""
        if not self.current_connection:
            return
        
        try:
            message = response.model_dump_json()
            await self.current_connection.send_text(message)
            
            if Config.VERBOSE and response.type in ["partial", "final"]:
                LOGGER.info(f"Sent {response.type}: {response.text}")
        
        except Exception as e:
            LOGGER.error(f"Error sending message: {e}")
            await self.disconnect()
    
    async def _send_error(self, error_message: str):
        """Send error message to client"""
        await self._send_message(TranscriptionResponse(
            type="error",
            timestamp=datetime.now().isoformat(),
            error_message=error_message
        ))
    
    def _reset_audio_state(self):
        """Reset all audio processing state"""
        self.audio_buffer.clear()
        self.is_speaking = False
        self.speech_frames.clear()
        self.silence_count = 0
        self.last_partial_time = 0.0
        
        if Config.VERBOSE:
            LOGGER.info("Audio state reset")
    
    def get_status(self) -> dict:
        """Get current connection status"""
        if not self.connection_status:
            return {"connected": False}
        
        return {
            "connected": True,
            "client_id": self.connection_status.client_id,
            "connected_at": self.connection_status.connected_at.isoformat(),
            "is_speaking": self.connection_status.is_speaking,
            "frames_processed": self.connection_status.frames_processed,
            "buffer_size": len(self.audio_buffer),
            "speech_frames": len(self.speech_frames)
        }