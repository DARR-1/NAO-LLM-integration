import asyncio
import json
import traceback
import wave
import os
import websockets
from interaction_proccesor import chat_stream, transcribe_audio_from_path, VADDetector

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1

USER_ID = "user_123"  # Default user ID if not set by the client
GRACE_SECONDS = 0.6  # ventana de gracia: si el usuario retoma a hablar dentro
                      # de este tiempo tras el corte, cancelamos la respuesta
                      # en curso y seguimos el mismo turno


def guardar_pcm_como_wav(pcm_bytes: bytes, path: str):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)


async def nao_handler(websocket):
    print("NAO conectado")

    audio_buffer = b""
    resume_buffer = b""
    user_id = USER_ID
    vad = VADDetector()
    loop = asyncio.get_event_loop()

    pending_task = None      # Task con la respuesta en curso (o None)
    pending_deadline = None  # loop.time() hasta el que esperamos un "resumed"

    try:
        async for message in websocket:

            # ==========================================
            # BYTES = CHUNK DE AUDIO (llega SIEMPRE, sin parar)
            # ==========================================
            if isinstance(message, bytes):

                # Si hay una respuesta en curso, estamos en la ventana de
                # gracia: seguimos alimentando el VAD para ver si el usuario
                # retoma a hablar antes de dar el turno por cerrado.
                if pending_task is not None:
                    estado = await asyncio.to_thread(vad.process_chunk, message)
                    resume_buffer += message

                    if estado == "resumed":
                        print("El usuario retomó a hablar, cancelando la respuesta en curso")
                        pending_task.cancel()
                        pending_task = None
                        audio_buffer = resume_buffer
                        resume_buffer = b""
                        continue

                    if loop.time() < pending_deadline:
                        continue  # seguimos dentro de la ventana de gracia

                    # Se acabó la ventana de gracia sin que retome: cerramos el turno
                    pending_task = None
                    resume_buffer = b""
                    vad.reset()
                    continue

                # Corremos el VAD en un hilo aparte: usa torch, es CPU-bound,
                # y no queremos bloquear el event loop en cada chunkcito.
                estado = await asyncio.to_thread(vad.process_chunk, message)

                if estado == "silence":
                    # Todavía no ha empezado a hablar, no acumulamos nada
                    continue

                # Si ya está hablando (o acaba de terminar), guardamos el chunk
                audio_buffer += message

                if estado == "speaking":
                    continue  # sigue hablando, seguimos esperando más chunks

                if estado == "turn_ended":
                    print("Fin de turno detectado (silencio tras hablar)")

                    if not audio_buffer:
                        vad.reset()
                        continue

                    # Lanzamos el procesamiento como tarea aparte para no
                    # bloquear la recepción de audio: así, si el usuario
                    # retoma a hablar dentro de la ventana de gracia, la
                    # podemos cancelar y seguir con el mismo turno.
                    pending_task = asyncio.create_task(
                        _process_turn(websocket, user_id, audio_buffer)
                    )
                    pending_deadline = loop.time() + GRACE_SECONDS
                    audio_buffer = b""
                    resume_buffer = b""

                continue

            # ==========================================
            # TEXTO = JSON (solo para setear user_id o chat de prueba)
            # ==========================================
            try:
                data = json.loads(message)
            except ValueError:
                print("Mensaje desconocido:", message)
                continue

            message_type = data.get("type")

            if message_type == "set_user":
                user_id = data.get("user", user_id)
                print("Usuario seteado:", user_id)
                continue

            if message_type == "chat":
                user_id = data.get("user", "desconocido")
                mensaje = data.get("message", "")
                if not mensaje:
                    continue

                print("Mensaje recibido:", mensaje)

                async for chunk in _stream_llm_response(user_id, mensaje):
                    await websocket.send(json.dumps({"type": "response_chunk", "text": chunk}))

                await websocket.send(json.dumps({"type": "response_end"}))

    except websockets.exceptions.ConnectionClosed as e:
        print("NAO desconectado. Code: %s, Reason: %s" % (e.code, e.reason))
    except Exception:
        print("¡ERROR en el handler!")
        traceback.print_exc()
    finally:
        if pending_task is not None:
            pending_task.cancel()


async def _process_turn(websocket, user_id, audio_bytes):
    """
    Transcribe el audio del turno, genera la respuesta con el LLM y la manda
    al cliente. Corre como tarea aparte del loop principal para poder
    cancelarse si el usuario retoma a hablar dentro de la ventana de gracia
    (ver GRACE_SECONDS en nao_handler).
    """
    temp_wav = "temp_audio_%s.wav" % user_id
    guardar_pcm_como_wav(audio_bytes, temp_wav)

    try:
        print("Transcribiendo...")
        transcription = await asyncio.to_thread(transcribe_audio_from_path, temp_wav)
        print("Transcripción:", transcription)

        if not transcription.strip():
            print("Transcripción vacía, ignorando turno")
            return

        async for chunk in _stream_llm_response(user_id, transcription):
            print("Mandando chunk al cliente:", chunk)
            await websocket.send(json.dumps({"type": "response_chunk", "text": chunk}))

        await websocket.send(json.dumps({"type": "response_end"}))

    except asyncio.CancelledError:
        print("Respuesta cancelada: el usuario retomó a hablar")
        raise
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)


async def _stream_llm_response(user_id, mensaje):
    loop = asyncio.get_event_loop()
    gen = chat_stream(user_id, mensaje)
    while True:
        chunk = await loop.run_in_executor(None, lambda: next(gen, None))
        if chunk is None:
            break
        yield chunk


async def main():
    print("Servidor WebSocket iniciado en puerto 8765")
    async with websockets.serve(nao_handler, "0.0.0.0", 8765, ping_interval=None):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())