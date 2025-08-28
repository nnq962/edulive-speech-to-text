import asyncio
import websockets
import json
import pyaudio
import numpy as np
from datetime import datetime

async def debug_client():
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
        
        print("=" * 60)
        print("🔍 RAW WebSocket Messages Debug Client")
        print("=" * 60)
        print("Connecting to STT server...")
        
        async with websockets.connect(uri) as websocket:
            print("✅ Connected! Starting audio stream...")
            print("📨 Raw JSON messages will be shown below:")
            print("-" * 60)
            
            # Send start message
            start_msg = json.dumps({"action": "start"})
            await websocket.send(start_msg)
            print(f"📤 SENT: {start_msg}")
            
            # Start audio streaming
            async def send_audio():
                try:
                    chunk_count = 0
                    while True:
                        data = stream.read(CHUNK, exception_on_overflow=False)
                        await websocket.send(data)
                        
                        chunk_count += 1
                        if chunk_count % 100 == 0:  # Every ~3 seconds
                            print(f"📡 Sent {chunk_count} audio chunks ({chunk_count * 30}ms)")
                        
                        await asyncio.sleep(0.03)  # ~30ms
                except Exception as e:
                    print(f"❌ Audio streaming error: {e}")
            
            # Handle server messages
            async def receive_messages():
                try:
                    message_count = 0
                    async for raw_message in websocket:
                        message_count += 1
                        
                        print(f"\n📨 MESSAGE #{message_count} - Raw JSON:")
                        print("─" * 40)
                        
                        # Pretty print the raw JSON
                        try:
                            data = json.loads(raw_message)
                            formatted_json = json.dumps(data, indent=2, ensure_ascii=False)
                            print(formatted_json)
                        except json.JSONDecodeError:
                            print(f"❌ Invalid JSON: {raw_message}")
                        
                        print("─" * 40)
                        
                        # Also show parsed version for readability
                        if isinstance(data, dict):
                            msg_type = data.get("type", "unknown")
                            timestamp = data.get("timestamp", "no-timestamp")
                            
                            if msg_type == "partial":
                                print(f"🔄 PARSED: [{timestamp}] PARTIAL: '{data.get('text', '')}'")
                                print(f"   Confidence: {data.get('confidence', 'N/A')}")
                            elif msg_type == "final":
                                print(f"✅ PARSED: [{timestamp}] FINAL: '{data.get('text', '')}'")
                                print(f"   Confidence: {data.get('confidence', 'N/A')}")
                            elif msg_type == "speech_start":
                                print(f"🎤 PARSED: [{timestamp}] Speech started")
                            elif msg_type == "speech_end":
                                print(f"🔇 PARSED: [{timestamp}] Speech ended")
                            elif msg_type == "status":
                                print(f"ℹ️  PARSED: [{timestamp}] Status: {data.get('text', '')}")
                            elif msg_type == "error":
                                print(f"❌ PARSED: [{timestamp}] Error: {data.get('error_message', '')}")
                            else:
                                print(f"❓ PARSED: [{timestamp}] Unknown type: {msg_type}")
                        
                        print("=" * 60)
                            
                except Exception as e:
                    print(f"❌ Message receiving error: {e}")
            
            # Run both tasks concurrently
            await asyncio.gather(
                send_audio(),
                receive_messages()
            )
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
    finally:
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        audio.terminate()

async def test_with_generated_audio():
    """Test với generated audio để show message format"""
    uri = "ws://localhost:8808/ws"
    
    print("=" * 60)
    print("🔍 Generated Audio Test - Raw Messages")
    print("=" * 60)
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected! Sending test audio...")
            
            # Send start message
            start_msg = json.dumps({"action": "start"})
            await websocket.send(start_msg)
            print(f"📤 SENT: {start_msg}")
            
            # Generate test audio (3 seconds of 440Hz sine wave)
            sample_rate = 16000
            duration = 3.0
            frequency = 440
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            sine_wave = np.sin(2 * np.pi * frequency * t)
            audio_data = (sine_wave * 16000).astype(np.int16).tobytes()  # Higher amplitude
            
            # Send in chunks
            chunk_size = 512
            print(f"📡 Sending {len(audio_data)} bytes in {chunk_size}-byte chunks...")
            
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                await websocket.send(chunk)
                await asyncio.sleep(0.03)
            
            print("📡 Audio sent, waiting for responses...")
            
            # Listen for responses with timeout
            message_count = 0
            timeout = 10
            
            try:
                async for raw_message in asyncio.wait_for(websocket, timeout=timeout):
                    message_count += 1
                    
                    print(f"\n📨 MESSAGE #{message_count} - Raw JSON:")
                    print("─" * 40)
                    
                    try:
                        data = json.loads(raw_message)
                        formatted_json = json.dumps(data, indent=2, ensure_ascii=False)
                        print(formatted_json)
                    except json.JSONDecodeError:
                        print(f"❌ Invalid JSON: {raw_message}")
                    
                    print("─" * 40)
            except asyncio.TimeoutError:
                print(f"⏰ Timeout after {timeout} seconds")
                
    except Exception as e:
        print(f"❌ Connection error: {e}")

if __name__ == "__main__":
    import sys
    
    print("Real-time STT Debug Client")
    print("Choose mode:")
    print("1. Live microphone (default)")
    print("2. Generated test audio")
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Running test mode...")
        try:
            asyncio.run(test_with_generated_audio())
        except KeyboardInterrupt:
            print("\nTest stopped")
    else:
        print("Running live microphone mode...")
        print("Press Ctrl+C to exit")
        try:
            asyncio.run(debug_client())
        except KeyboardInterrupt:
            print("\nClient stopped")