# Edulive Speech-to-Text Service

Dịch vụ Speech-to-Text thời gian thực sử dụng Whisper với hỗ trợ WebSocket để nhận dạng giọng nói tiếng Việt.

## Tổng quan

Service này cung cấp khả năng nhận dạng giọng nói thời gian thực thông qua WebSocket, hỗ trợ cả transcription tạm thời (partial) và cuối cùng (final) với độ chính xác cao.

## Tính năng chính

- **Real-time transcription**: Nhận dạng giọng nói thời gian thực
- **Voice Activity Detection (VAD)**: Tự động phát hiện khi có/không có giọng nói
- **Dual model system**: Model nhỏ cho partial transcription, model lớn cho final transcription
- **Hallucination suppression**: Lọc bỏ các câu ảo tưởng phổ biến
- **WebSocket support**: Kết nối WebSocket ổn định với xử lý lỗi

## API Endpoints

### 1. Health Check Endpoints

#### GET `/`
**Mô tả**: Endpoint gốc để kiểm tra trạng thái service

**Response thành công (200)**:
```json
{
  "message": "Real-time STT Service",
  "version": "1.0.0", 
  "status": "running",
  "config": {
    "device": "cuda",
    "compute_type": "float16",
    "partial_model": "small",
    "final_model": "medium",
    "language": "vi",
    "silence_threshold": 800,
    "verbose": false
  }
}
```

#### GET `/status`
**Mô tả**: Kiểm tra trạng thái chi tiết của service và connection

**Response thành công (200)**:
```json
{
  "service": "running",
  "connection": {
    "connected": true,
    "client_id": "uuid-string",
    "connected_at": "2024-01-01T10:00:00.000Z",
    "is_speaking": false,
    "frames_processed": 1234,
    "buffer_size": 10,
    "speech_frames": 0
  },
  "config": {
    "device": "cuda",
    "compute_type": "float16",
    "partial_model": "small", 
    "final_model": "medium",
    "language": "vi",
    "silence_threshold": 800,
    "verbose": false
  }
}
```

**Response lỗi (503)**:
```json
{
  "error": "Service not initialized"
}
```

### 2. WebSocket Endpoint

#### WS `/ws`
**Mô tả**: WebSocket endpoint cho audio streaming và nhận kết quả transcription

## WebSocket Protocol

### Kết nối WebSocket

1. **Kết nối thành công**:
   - Client nhận được welcome message
   - Chỉ cho phép 1 connection đồng thời
   - Connection mới sẽ thay thế connection cũ

2. **Kết nối thất bại**:
   - WebSocket đóng với code `1011` và reason "Service not initialized"

### Messages từ Client

#### 1. Control Messages (JSON)

**Start recording**:
```json
{
  "action": "start"
}
```

**Stop recording**:
```json
{
  "action": "stop"
}
```

#### 2. Audio Data (Binary)
- Gửi audio data dưới dạng binary frames
- Format: PCM 16-bit, 16kHz sample rate
- Mỗi frame nên khoảng 20ms audio

### Messages từ Server

Service trả về các loại response sau thông qua WebSocket:

#### 1. Status Messages

**Welcome message** (khi kết nối):
```json
{
  "type": "status",
  "text": "Connected to STT service",
  "timestamp": "2024-01-01T10:00:00.000Z"
}
```

**Recording started**:
```json
{
  "type": "status", 
  "text": "Ready to receive audio",
  "timestamp": "2024-01-01T10:00:00.000Z"
}
```

**Recording stopped**:
```json
{
  "type": "status",
  "text": "Audio recording stopped", 
  "timestamp": "2024-01-01T10:00:00.000Z"
}
```

**Connection replacement**:
```json
{
  "type": "status",
  "text": "Connection replaced by new client",
  "timestamp": "2024-01-01T10:00:00.000Z"
}
```

#### 2. Speech Detection Events

**Speech started**:
```json
{
  "type": "speech_start",
  "timestamp": "2024-01-01T10:00:00.000Z"
}
```

**Speech ended**:
```json
{
  "type": "speech_end", 
  "timestamp": "2024-01-01T10:00:00.000Z"
}
```

#### 3. Transcription Results

**Partial transcription** (trong lúc nói):
```json
{
  "type": "partial",
  "text": "xin chào tôi đang",
  "timestamp": "2024-01-01T10:00:01.500Z",
  "confidence": 0.85
}
```

**Final transcription** (khi kết thúc câu):
```json
{
  "type": "final",
  "text": "xin chào tôi đang nói tiếng Việt",
  "timestamp": "2024-01-01T10:00:05.000Z", 
  "confidence": 0.92
}
```

**Empty transcription** (không nhận dạng được/bị lọc):
```json
{
  "type": "partial",
  "text": "",
  "timestamp": "2024-01-01T10:00:02.000Z",
  "confidence": 0.0
}
```

#### 4. Error Messages

**Invalid audio data**:
```json
{
  "type": "error",
  "timestamp": "2024-01-01T10:00:00.000Z",
  "error_message": "Invalid audio data"
}
```

**Transcription timeout**:
```json
{
  "type": "error", 
  "timestamp": "2024-01-01T10:00:00.000Z",
  "error_message": "Partial transcription failed: Partial transcription timeout"
}
```

**Processing error**:
```json
{
  "type": "error",
  "timestamp": "2024-01-01T10:00:00.000Z", 
  "error_message": "Message processing error"
}
```

**Model error**:
```json
{
  "type": "error",
  "timestamp": "2024-01-01T10:00:00.000Z",
  "error_message": "Final transcription failed: [error details]"
}
```

## Luồng hoạt động

### 1. Kết nối và khởi tạo
```
Client → WS Connect → Server
Client ← Welcome Message ← Server  
Client → {"action": "start"} → Server
Client ← Status "Ready" ← Server
```

### 2. Speech Detection và Transcription
```
Client → Audio Data → Server (VAD Detection)
Client ← Speech Start ← Server
Client → More Audio → Server
Client ← Partial Results ← Server (every 1.5s)
Client → Audio Data → Server (silence detected)
Client ← Final Result ← Server  
Client ← Speech End ← Server
```

### 3. Ngắt kết nối
```
Client → {"action": "stop"} → Server
Client ← Status "Stopped" ← Server
Client → WS Close → Server
```

## Error Handling

### Client-side Errors
- **Connection refused**: Service chưa khởi tạo
- **Connection replaced**: Client khác đã kết nối
- **WebSocket disconnect**: Mất kết nối mạng

### Server-side Errors  
- **Audio processing errors**: Audio data không hợp lệ
- **Model errors**: Lỗi từ Whisper model
- **Timeout errors**: Transcription quá thời gian cho phép
- **Resource errors**: Hết memory hoặc GPU

### Error Recovery
- Client nên reconnect khi bị disconnect
- Retry logic cho failed transcriptions
- Fallback mechanisms cho model errors

## Configuration

Service có thể config thông qua file `config.py`:

- **Models**: `small` (partial) / `medium` (final)  
- **Language**: `vi` (tiếng Việt)
- **Device**: Auto-detect CUDA/CPU
- **VAD threshold**: 800 (có thể điều chỉnh theo mic)
- **Timing**: Partial interval 1.5s, silence threshold 4s

## Requirements

- Python 3.8+
- CUDA (tùy chọn, cho GPU acceleration)
- faster-whisper
- FastAPI
- WebSocket support

## Usage Example

Xem file `test_client.py` và `debug_client.py` để có ví dụ về cách sử dụng WebSocket client.
