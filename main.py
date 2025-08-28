import asyncio
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from config import Config
from stt_service import STTService
from websocket_handler import WebSocketHandler
from utils.logger_config import LOGGER

# Global services
stt_service: STTService = None
websocket_handler: WebSocketHandler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global stt_service, websocket_handler
    
    try:
        LOGGER.info("Starting Real-time STT Service...")
        LOGGER.info(f"Configuration: {Config.log_config()}")
        
        # Initialize STT service
        stt_service = STTService()
        await stt_service.initialize_models()
        
        # Initialize WebSocket handler
        websocket_handler = WebSocketHandler(stt_service)
        
        LOGGER.info(f"Server ready on {Config.HOST}:{Config.PORT}")
        yield
        
    except Exception as e:
        LOGGER.error(f"Failed to start service: {e}")
        raise
    finally:
        # Cleanup
        if stt_service:
            await stt_service.cleanup()
        LOGGER.info("Service shutdown complete")

# FastAPI app
app = FastAPI(
    title="Real-time STT Service",
    description="Real-time Speech-to-Text service with WebSocket support",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Root endpoint"""
    return JSONResponse({
        "message": "Real-time STT Service",
        "version": "1.0.0",
        "status": "running",
        "config": Config.log_config()
    })

@app.get("/status")
async def get_status():
    """Get service status"""
    global websocket_handler
    
    if not websocket_handler:
        return JSONResponse({"error": "Service not initialized"}, status_code=503)
    
    return JSONResponse({
        "service": "running",
        "connection": websocket_handler.get_status(),
        "config": Config.log_config()
    })

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for audio streaming"""
    global websocket_handler
    
    if not websocket_handler:
        await websocket.close(code=1011, reason="Service not initialized")
        return
    
    client_id = str(uuid.uuid4())
    
    try:
        # Connect client
        await websocket_handler.connect(websocket, client_id)
        
        # Handle messages
        while True:
            try:
                message = await websocket.receive()
                
                if message["type"] == "websocket.receive":
                    if "bytes" in message:
                        await websocket_handler.handle_message(websocket, message["bytes"])
                    elif "text" in message:
                        await websocket_handler.handle_message(websocket, message["text"])
                
                elif message["type"] == "websocket.disconnect":
                    break
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                LOGGER.error(f"WebSocket message error: {e}")
                break
    
    except Exception as e:
        LOGGER.error(f"WebSocket connection error: {e}")
    
    finally:
        await websocket_handler.disconnect()

if __name__ == "__main__":
    import uvicorn
    
    # Log startup info
    LOGGER.info("=" * 50)
    LOGGER.info("Real-time STT Service")
    LOGGER.info("=" * 50)
    
    # Run server
    uvicorn.run(
        "main:app",
        host=Config.HOST,
        port=Config.PORT,
        log_level="info" if Config.VERBOSE else "warning",
        access_log=Config.VERBOSE,
        reload=False
    )