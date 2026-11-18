import asyncio
import wave
import os
import json
import io

import numpy as np
import librosa
import uvicorn

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.core.pipeline import DefensePipeline


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Voice Impersonation Defense API"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# AI PIPELINE
# ============================================================

pipeline = DefensePipeline(
    target_speaker_embedding=None
)


# ============================================================
# CONNECTION MANAGER
# ============================================================

class ConnectionManager:

    def __init__(self):

        self.monitors: list[WebSocket] = []

        self.calls: dict[str, list[WebSocket]] = {}


    async def connect(
        self,
        websocket: WebSocket,
        is_monitor: bool = False,
        call_id: str = None
    ):

        # Always accept the WebSocket.
        # The previous version rejected connections after
        # two connections and caused 403 errors during testing.

        await websocket.accept()


        # ----------------------------------------------------
        # Frontend monitoring connection
        # ----------------------------------------------------

        if is_monitor:

            self.monitors.append(websocket)

            print(
                "[WS] Monitor connected"
            )

            return True


        # ----------------------------------------------------
        # Normal call connection
        # ----------------------------------------------------

        if call_id not in self.calls:

            self.calls[call_id] = []


        self.calls[call_id].append(websocket)


        print(
            f"[WS] Client connected: "
            f"{call_id} "
            f"(connections={len(self.calls[call_id])})"
        )


        return True


    def disconnect(
        self,
        websocket: WebSocket,
        is_monitor: bool = False,
        call_id: str = None
    ):

        # ----------------------------------------------------
        # Monitor
        # ----------------------------------------------------

        if is_monitor:

            if websocket in self.monitors:

                self.monitors.remove(websocket)


            print(
                "[WS] Monitor disconnected"
            )

            return


        # ----------------------------------------------------
        # Normal client
        # ----------------------------------------------------

        if call_id in self.calls:

            if websocket in self.calls[call_id]:

                self.calls[call_id].remove(websocket)


            remaining = len(
                self.calls[call_id]
            )


            print(
                f"[WS] Client disconnected: "
                f"{call_id} "
                f"(remaining={remaining})"
            )


            # Remove empty call entries

            if remaining == 0:

                del self.calls[call_id]


    async def broadcast_result(
        self,
        data: dict,
        call_id: str
    ):

        # ----------------------------------------------------
        # Send result to frontend monitors
        # ----------------------------------------------------

        for monitor in list(self.monitors):

            try:

                await monitor.send_json(
                    data
                )

            except Exception:

                try:

                    self.monitors.remove(
                        monitor
                    )

                except ValueError:

                    pass


        # ----------------------------------------------------
        # Send result to call clients
        # ----------------------------------------------------

        if call_id not in self.calls:

            return


        for client in list(
            self.calls[call_id]
        ):

            try:

                await client.send_json(
                    data
                )

            except Exception:

                self.disconnect(
                    client,
                    is_monitor=False,
                    call_id=call_id
                )


    async def route_audio(
        self,
        sender: WebSocket,
        call_id: str,
        audio_bytes: bytes
    ):

        if call_id not in self.calls:

            return


        for client in list(
            self.calls[call_id]
        ):

            if client == sender:

                continue


            try:

                await client.send_bytes(
                    audio_bytes
                )

            except Exception:

                self.disconnect(
                    client,
                    is_monitor=False,
                    call_id=call_id
                )


# ============================================================
# GLOBAL CONNECTION MANAGER
# ============================================================

manager = ConnectionManager()


# ============================================================
# DEMO TASK STORAGE
#
# Keeps track of *888 / *999 demo streams so that an old
# stream can be cancelled when a call ends or a new demo starts.
# ============================================================

demo_tasks: dict[str, asyncio.Task] = {}


# ============================================================
# DEMO STREAM
#
# *999 -> scam_call.wav
# *888 -> human_call.wav
#
# The audio is processed by the SAME AI pipeline used by
# /api/analyze-audio.
# ============================================================

async def run_demo_stream(
    call_id: str,
    file_type: str
):

    filepath = (
        f"data/{file_type}_call.wav"
    )


    # --------------------------------------------------------
    # Validate demo type
    # --------------------------------------------------------

    if file_type not in (
        "scam",
        "human"
    ):

        print(
            f"[DEMO] Invalid demo type: "
            f"{file_type}"
        )

        return


    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if not os.path.exists(filepath):

        print(
            f"[DEMO] File missing: "
            f"{filepath}"
        )

        return


    print()
    print(
        "=" * 60
    )
    print(
        f"[DEMO] STARTING {file_type.upper()} DEMO"
    )
    print(
        f"[DEMO] Call ID : {call_id}"
    )
    print(
        f"[DEMO] File    : {filepath}"
    )
    print(
        "=" * 60
    )


    try:

        with wave.open(
            filepath,
            "rb"
        ) as wf:

            framerate = wf.getframerate()

            channels = wf.getnchannels()

            sample_width = wf.getsampwidth()


            print(
                f"[DEMO] Sample rate : "
                f"{framerate} Hz"
            )

            print(
                f"[DEMO] Channels    : "
                f"{channels}"
            )

            print(
                f"[DEMO] Bit depth   : "
                f"{sample_width * 8}"
            )


            # ------------------------------------------------
            # 500 ms chunks
            # ------------------------------------------------

            chunk_size = int(
                framerate * 0.5
            )


            chunk_number = 0


            while True:

                frames = wf.readframes(
                    chunk_size
                )


                if not frames:

                    break


                # ------------------------------------------------
                # Convert WAV PCM int16 -> float32
                # ------------------------------------------------

                audio_int16 = np.frombuffer(
                    frames,
                    dtype=np.int16
                )


                audio_float32 = (
                    audio_int16.astype(
                        np.float32
                    )
                    / 32768.0
                )


                # ------------------------------------------------
                # AI PROCESSING
                # ------------------------------------------------

                result = (
                    pipeline.process_stream_chunk(
                        audio_float32
                    )
                )


                chunk_number += 1


                status = result.get(
                    "status",
                    "unknown"
                )


                risk = result.get(
                    "risk_score",
                    0.0
                )


                print(
                    f"[DEMO] {call_id} | "
                    f"chunk={chunk_number} | "
                    f"status={status} | "
                    f"risk={risk}"
                )


                # ------------------------------------------------
                # Send result to mobile
                # ------------------------------------------------

                await manager.broadcast_result(
                    data=result,
                    call_id=call_id
                )


                # ------------------------------------------------
                # Simulate real-time streaming
                # ------------------------------------------------

                await asyncio.sleep(
                    0.5
                )


    except asyncio.CancelledError:

        print(
            f"[DEMO] CANCELLED: "
            f"{call_id}"
        )

        raise


    except Exception as e:

        print(
            f"[DEMO] ERROR for "
            f"{call_id}: "
            f"{type(e).__name__}: {e}"
        )


    finally:

        print(
            f"[DEMO] FINISHED: "
            f"{call_id}"
        )

        print(
            "=" * 60
        )


# ============================================================
# FILE ANALYSIS API
#
# Used by the Scan tab.
# ============================================================

@app.post(
    "/api/analyze-audio"
)
async def analyze_audio_file(
    file: UploadFile = File(...)
):

    try:

        print()
        print(
            f"[SCAN] Received file: "
            f"{file.filename}"
        )


        # ----------------------------------------------------
        # Read uploaded file
        # ----------------------------------------------------

        contents = await file.read()


        print(
            f"[SCAN] File size: "
            f"{len(contents)} bytes"
        )


        audio_stream = io.BytesIO(
            contents
        )


        # ----------------------------------------------------
        # Load and resample
        # ----------------------------------------------------

        audio_float32, sr = (
            librosa.load(
                audio_stream,
                sr=16000,
                mono=True
            )
        )


        print(
            f"[SCAN] Audio loaded: "
            f"{len(audio_float32)} samples "
            f"at {sr} Hz"
        )


        # ----------------------------------------------------
        # Process 500 ms chunks
        # ----------------------------------------------------

        chunk_size = 8000

        scores = []


        for i in range(
            0,
            len(audio_float32),
            chunk_size
        ):

            chunk = audio_float32[
                i:i + chunk_size
            ]


            # Ignore fragments under 100 ms

            if len(chunk) < 1600:

                continue


            result = (
                pipeline.process_stream_chunk(
                    chunk
                )
            )


            if (
                result.get("status")
                == "speech_detected"
            ):

                scores.append(
                    result.get(
                        "risk_score",
                        0.0
                    )
                )


        # ----------------------------------------------------
        # Calculate final score
        # ----------------------------------------------------

        if scores:

            final_risk = round(
                float(
                    np.mean(scores)
                ),
                2
            )


            peak_risk = round(
                float(
                    np.max(scores)
                ),
                2
            )


            # Preserve your previous scoring behaviour

            reported_score = (
                peak_risk
                if peak_risk > 60
                else final_risk
            )

        else:

            reported_score = 0.0


        # ----------------------------------------------------
        # Alert level
        # ----------------------------------------------------

        alert_level = "Safe"


        if reported_score >= 75.0:

            alert_level = "Critical"

        elif reported_score >= 45.0:

            alert_level = "Suspicious"


        print(
            f"[SCAN] Final risk: "
            f"{reported_score}"
        )

        print(
            f"[SCAN] Alert: "
            f"{alert_level}"
        )

        print(
            f"[SCAN] Chunks: "
            f"{len(scores)}"
        )


        return {

            "success": True,

            "filename": file.filename,

            "risk_score": reported_score,

            "alert_level": alert_level,

            "chunks_analyzed": len(scores)

        }


    except Exception as e:

        print(
            f"[SCAN] ERROR: "
            f"{type(e).__name__}: {e}"
        )


        return {

            "success": False,

            "error": str(e),

            "risk_score": 0.0

        }


# ============================================================
# WEBSOCKET
#
# /ws/stream/*999
# /ws/stream/*888
#
# Used for real-time AI demo scoring.
# ============================================================

@app.websocket(
    "/ws/stream/{call_id}"
)
async def websocket_endpoint(
    websocket: WebSocket,
    call_id: str
):

    print()
    print(
        "=" * 60
    )

    print(
        "[WS] CONNECTION REQUEST"
    )

    print(
        f"[WS] call_id = {call_id}"
    )

    print(
        f"[WS] client  = {websocket.client}"
    )

    print(
        "=" * 60
    )


    # ========================================================
    # FRONTEND MONITOR
    # ========================================================

    if call_id == "frontend_monitor":

        await manager.connect(
            websocket,
            is_monitor=True
        )


        try:

            while True:

                await websocket.receive_text()


        except WebSocketDisconnect:

            manager.disconnect(
                websocket,
                is_monitor=True
            )


        except Exception as e:

            print(
                f"[WS] Monitor error: "
                f"{type(e).__name__}: {e}"
            )


            manager.disconnect(
                websocket,
                is_monitor=True
            )


        return


    # ========================================================
    # NORMAL CALL
    # ========================================================

    await manager.connect(
        websocket,
        is_monitor=False,
        call_id=call_id
    )


    print(
        f"[WS] Connected successfully: "
        f"{call_id}"
    )


    try:

        while True:

            message = await websocket.receive()


            # =================================================
            # TEXT MESSAGE
            # =================================================

            if message.get("text") is not None:

                try:

                    command = json.loads(
                        message["text"]
                    )


                    print(
                        f"[WS] Command from "
                        f"{call_id}: "
                        f"{command}"
                    )


                    # -------------------------------------------------
                    # START DEMO
                    # -------------------------------------------------

                    if (
                        command.get("action")
                        == "start_demo"
                    ):

                        demo_type = (
                            command.get("type")
                        )


                        if demo_type not in (
                            "scam",
                            "human"
                        ):

                            print(
                                f"[WS] Invalid "
                                f"demo type: "
                                f"{demo_type}"
                            )

                            continue


                        # ---------------------------------------------
                        # Cancel an existing demo for this call
                        # ---------------------------------------------

                        if call_id in demo_tasks:

                            old_task = (
                                demo_tasks[call_id]
                            )


                            if not old_task.done():

                                print(
                                    f"[WS] Cancelling "
                                    f"old demo: "
                                    f"{call_id}"
                                )


                                old_task.cancel()


                                try:

                                    await old_task

                                except asyncio.CancelledError:

                                    pass


                        # ---------------------------------------------
                        # Start new demo
                        # ---------------------------------------------

                        print(
                            f"[WS] Starting "
                            f"{demo_type} demo "
                            f"for {call_id}"
                        )


                        task = asyncio.create_task(
                            run_demo_stream(
                                call_id,
                                demo_type
                            )
                        )


                        demo_tasks[
                            call_id
                        ] = task


                except json.JSONDecodeError as e:

                    print(
                        f"[WS] Invalid JSON: "
                        f"{e}"
                    )


            # =================================================
            # BINARY AUDIO
            # =================================================

            elif message.get("bytes") is not None:

                data = message["bytes"]


                print(
                    f"[WS] Audio received: "
                    f"{len(data)} bytes"
                )


                # -------------------------------------------------
                # Route audio to other connected clients
                # -------------------------------------------------

                await manager.route_audio(
                    sender=websocket,
                    call_id=call_id,
                    audio_bytes=data
                )


                # -------------------------------------------------
                # Convert PCM audio
                # -------------------------------------------------

                audio_int16 = np.frombuffer(
                    data,
                    dtype=np.int16
                )


                audio_float32 = (
                    audio_int16.astype(
                        np.float32
                    )
                    / 32768.0
                )


                # -------------------------------------------------
                # AI analysis
                # -------------------------------------------------

                result = (
                    pipeline.process_stream_chunk(
                        audio_float32
                    )
                )


                print(
                    f"[WS] Live result: "
                    f"status="
                    f"{result.get('status')} "
                    f"risk="
                    f"{result.get('risk_score')}"
                )


                # -------------------------------------------------
                # Broadcast AI result
                # -------------------------------------------------

                await manager.broadcast_result(
                    data=result,
                    call_id=call_id
                )


    except WebSocketDisconnect:

        print(
            f"[WS] Client disconnected: "
            f"{call_id}"
        )


    except Exception as e:

        print(
            f"[WS] ERROR for {call_id}: "
            f"{type(e).__name__}: {e}"
        )


    finally:

        # ========================================================
        # CANCEL DEMO TASK
        # ========================================================

        if call_id in demo_tasks:

            task = demo_tasks[
                call_id
            ]


            if not task.done():

                print(
                    f"[WS] Cancelling demo "
                    f"task: {call_id}"
                )


                task.cancel()


                try:

                    await task

                except asyncio.CancelledError:

                    pass


            del demo_tasks[
                call_id
            ]


        # ========================================================
        # DISCONNECT CLIENT
        # ========================================================

        manager.disconnect(
            websocket,
            is_monitor=False,
            call_id=call_id
        )


        print(
            f"[WS] Cleanup complete: "
            f"{call_id}"
        )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
