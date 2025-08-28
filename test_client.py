import asyncio
import websockets
import json
import pyaudio
import numpy as np
from datetime import datetime

async def test_client():
    uri = "ws://localhost:8808/ws"
    
    # Audio settings
    RATE = 16000
    CHUNK = 512  # ~30ms
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    
    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    
    try:
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        print("Connecting to STT server...")
        async with websockets.connect(uri) as websocket:
            print("Connected! Starting audio stream...")
            
            # Send start message
            await websocket.send(json.dumps({"action": "start"}))
            
            # Start audio streaming
            async def send_audio():
                try:
                    while True:
                        data = stream.read(CHUNK, exception_on_overflow=False)
                        await websocket.send(data)
                        await asyncio.sleep(0.03)  # ~30ms
                except Exception as e:
                    print(f"Audio streaming error: {e}")
            
            # Handle server messages
            async def receive_messages():
                try:
                    async for message in websocket:
                        data = json.loads(message)
                        timestamp = data.get("timestamp", "")
                        msg_type = data.get("type")
                        
                        if msg_type == "partial":
                            print(f"🔄 [{timestamp}] PARTIAL: {data['text']}")
                        elif msg_type == "final":
                            print(f"✅ [{timestamp}] FINAL: {data['text']}")
                        elif msg_type == "speech_start":
                            print(f"🎤 [{timestamp}] Speech started")
                        elif msg_type == "speech_end":
                            print(f"🔇 [{timestamp}] Speech ended")
                        elif msg_type == "status":
                            print(f"ℹ️  [{timestamp}] {data['text']}")
                        elif msg_type == "error":
                            print(f"❌ [{timestamp}] ERROR: {data['error_message']}")
                        else:
                            print(f"📨 [{timestamp}] {data}")
                            
                except Exception as e:
                    print(f"Message receiving error: {e}")
            
            # Run both tasks concurrently
            await asyncio.gather(
                send_audio(),
                receive_messages()
            )
            
    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        audio.terminate()

if __name__ == "__main__":
    print("Real-time STT Test Client")
    print("Press Ctrl+C to exit")
    try:
        asyncio.run(test_client())
    except KeyboardInterrupt:
        print("\nClient stopped")