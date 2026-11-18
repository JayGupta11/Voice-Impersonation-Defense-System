# scripts/mock_call.py
import asyncio
import websockets
import json
import time
import librosa
import numpy as np

async def stream_audio():
    uri = "ws://localhost:8000/ws/stream/call_123"
    wav_file_path = "test_audio.wav"
    
    try:
        print(f"Loading and resampling {wav_file_path} to 16kHz...")
        audio_float32, sr = librosa.load(wav_file_path, sr=16000, mono=True)
        chunk_size = int(16000 * 0.25)
        
        async with websockets.connect(uri) as websocket:
            print("Connected to AI Defense API. Streaming looping audio...\n")
            
            while True:  # Loop the audio indefinitely for UI testing
                for i in range(0, len(audio_float32), chunk_size):
                    chunk = audio_float32[i:i + chunk_size]
                    if len(chunk) < chunk_size:
                        chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
                    
                    audio_int16 = (chunk * 32767).astype(np.int16)
                    
                    start_time = time.time()
                    await websocket.send(audio_int16.tobytes())
                    
                    response = await websocket.recv()
                    latency = (time.time() - start_time) * 1000
                    
                    data_json = json.loads(response)
                    print(f"Status: {data_json.get('status')} | Risk: {data_json.get('risk_score', 0)}")
                    
                    await asyncio.sleep(0.25)
                
    except FileNotFoundError:
        print(f"Error: Could not find {wav_file_path}.")
    except ConnectionRefusedError:
        print("Connection failed. Is the FastAPI server running?")

if __name__ == "__main__":
    asyncio.run(stream_audio())
