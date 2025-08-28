from pydantic import BaseModel
from typing import Literal, Optional
from datetime import datetime

class AudioMessage(BaseModel):
    """Audio data message from client"""
    action: Literal["start", "stop", "audio"]
    data: Optional[bytes] = None

class TranscriptionResponse(BaseModel):
    """Response message to client"""
    type: Literal["partial", "final", "speech_start", "speech_end", "error", "status"]
    text: Optional[str] = None
    timestamp: str
    confidence: Optional[float] = None
    error_message: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ConnectionStatus(BaseModel):
    """Connection status tracking"""
    client_id: str
    connected_at: datetime
    is_speaking: bool = False
    frames_processed: int = 0