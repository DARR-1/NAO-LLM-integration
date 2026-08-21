import asyncio
import json
import traceback
from unittest import case
import wave
import os
import websockets
from interaction_proccesor import chat_stream, transcribe_audio_from_path, VADDetector, sanitize_text

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1

USER_ID = "Crescencio"  # Usuario default
OFFSET_SECONDS = 0.6  # si el usuario retoma a hablar dentro
                      # de este tiempo tras el corte, cancelamos la respuesta
                      # en curso y seguimos el mismo turno
COOLDOWN_SECONDS = 1


def guardar_pcm_como_wav(pcm_bytes: bytes, path: str):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)

# ---------------------------------------------------------------------------
# MANEJO DE LA CONEXIÓN CON EL ROBOT NAO A TRAVÉS DE WEBSOCKET
# ---------------------------------------------------------------------------

async def nao_handler(websocket):
    print("NAO conectado")

    # Inicializamos buffers y VAD
    audio_buffer = b""
    resume_buffer = b""
    user_id = USER_ID
    vad = VADDetector()
    loop = asyncio.get_event_loop()

    pending_task = None      # Task con la respuesta en curso
    pending_deadline = None  # loop.time() hasta el que esperamos un "resumed"

    is_talking = False  # Flag para saber si NAO está hablando
    listen_after = 0.0   # timestamp (loop.time()) hasta el que ignoramos audio
                          # entrante tras el cooldown post-habla de NAO

    try:
        async for message in websocket:

            # Se activa cuando recibe datos en bytes (audio PCM) y NAO no está hablando
            if isinstance(message, bytes) and not is_talking:

                if loop.time() < listen_after:
                    # Todavía dentro de la ventana de cooldown post-habla:
                    # descartamos el chunk sin pasarlo por el VAD (evita que
                    # NAO se escuche a sí mismo por la cola/eco del TTS).
                    continue

                # Si hay una respuesta en curso, estamos en el offset
                # de espera: seguimos alimentando el VAD para ver si
                # el usuario retoma a hablar antes de dar por cerrado 
                # el turno.
                if pending_task is not None and not is_talking:
                    # Espera a que el usuario retome a hablar dentro del OFFSET_SECONDS
                    estado = await asyncio.to_thread(vad.process_chunk, message)
                    resume_buffer += message

                    if estado == "resumed" and not is_talking:
                        print("El usuario retomó a hablar, cancelando la respuesta en curso")
                        pending_task.cancel()
                        pending_task = None
                        audio_buffer = resume_buffer
                        resume_buffer = b""
                        continue

                    if loop.time() < pending_deadline and not is_talking:
                        continue  # seguimos esperando a ver si retoma a hablar

                    # Si llegamos acá, el usuario NO retomó a hablar
                    pending_task = None
                    resume_buffer = b""
                    vad.reset()
                    continue

                # Corremos el VAD en un hilo aparte: usa torch, es CPU-bound,
                # y no queremos bloquear el event loop en cada chunk.
                estado = await asyncio.to_thread(vad.process_chunk, message)

                if estado == "silence" and not is_talking:
                    # No se ha detectado voz, seguimos esperando más chunks
                    continue

                # Almacenamos los chunks en el buffer de audio
                audio_buffer += message

                if estado == "speaking":
                    continue  # Mientras siga hablando, seguimos acumulando audio

                if estado == "turn_ended" or is_talking:
                    print("Fin de turno detectado (silencio tras hablar)")

                    if not audio_buffer:
                        vad.reset()
                        continue

                    # Lanzamos el procesamiento como tarea aparte para no
                    # bloquear la recepción de audio: así, si el usuario
                    # retoma a hablar dentro de la ventana de espera, la
                    # podemos cancelar y seguir con el mismo turno.
                    pending_task = asyncio.create_task(
                        process_turn(websocket, user_id, audio_buffer)
                    )
                    pending_deadline = loop.time() + OFFSET_SECONDS
                    audio_buffer = b""
                    resume_buffer = b""

                continue

            # ==========================================
            # TEXTO = JSON
            # ==========================================
            try:
                data = json.loads(message)
            except ValueError:
                print("Mensaje desconocido:", message)
                continue

            message_type = data.get("type")

            match message_type:
                case "nao_speech_start":
                    is_talking = True
                    audio_buffer = b""
                    resume_buffer = b""
                    if pending_task is not None:
                        pending_task.cancel()
                        pending_task = None
                    await asyncio.to_thread(vad.reset)
                    print("NAO empezó a hablar")

                case "nao_speech_end":
                    is_talking = False
                    listen_after = loop.time() + COOLDOWN_SECONDS
                    await asyncio.to_thread(vad.reset)
                    print("NAO terminó de hablar, cooldown activo")

                case "set_user":
                    user_id = data.get("user", user_id)
                    print("Usuario seteado:", user_id)

                case "chat":
                    user_id = data.get("user", "desconocido")
                    mensaje = data.get("message", "")
                    if not mensaje:
                        continue
                    print("Mensaje recibido:", mensaje)

                    print("Generando respuesta con LLM...")
                    async for chunk in stream_llm_response(user_id, mensaje):
                        await websocket.send(json.dumps({"type": "response_chunk", "text": chunk}))

                    print("Respuesta enviada")
                    await websocket.send(json.dumps({"type": "response_end"}))

    except websockets.exceptions.ConnectionClosed as e:
        print("NAO desconectado. Code: %s, Reason: %s" % (e.code, e.reason))
    except Exception:
        print("¡ERROR en el handler!")
        traceback.print_exc()
    finally:
        if pending_task is not None:
            pending_task.cancel()


async def process_turn(websocket, user_id, audio_bytes):
    """
    Transcribe el audio del turno, genera la respuesta con el LLM y la manda
    al cliente. Corre como tarea aparte del loop principal para poder
    cancelarse si el usuario retoma a hablar dentro de la ventana de tiempo.
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

        async for chunk in stream_llm_response(user_id, transcription):
            #print("Mandando chunk al cliente:", chunk)
            await websocket.send(json.dumps({"type": "response_chunk", "text": chunk}))

        await websocket.send(json.dumps({"type": "response_end"}))

    except asyncio.CancelledError:
        print("Respuesta cancelada: el usuario retomó a hablar")
        raise
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)


async def stream_llm_response(user_id, mensaje):
    loop = asyncio.get_event_loop()
    gen = chat_stream(user_id, mensaje)
    while True:
        chunk = await loop.run_in_executor(None, lambda: next(gen, None))
        if chunk is None:
            break
        yield sanitize_text(chunk)


async def main():
    print("Servidor WebSocket iniciado en puerto 8765")
    async with websockets.serve(nao_handler, "0.0.0.0", 8765, ping_interval=None):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())