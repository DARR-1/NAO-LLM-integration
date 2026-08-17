"""
Maneja la conversación de NAO con un LLM local (Ollama), guardando
un history por usuario en archivos CSV para poder retomar la
conversación la próxima vez que hable con esa misma persona.
"""

import ollama
import csv
import os
from datetime import datetime
from faster_whisper import WhisperModel, BatchedInferencePipeline
from silero_vad import load_silero_vad

# ---------------------------------------------------------------------------
# CONFIGURACIÓN DE LA PERSONALIDAD DE NAO
# ---------------------------------------------------------------------------

NAME = "Artemis"  # nombre del robot

SYSTEM_PROMPT = f"""You are a NAO robot named {NAME}. You are a small,
friendly, and curious humanoid robot that interacts with people in an
educational/social environment.

Behavior rules:
- Always respond in Spanish, briefly and naturally (1-3 sentences), as if
  you were speaking out loud, not writing a long text.
- Your tone is warm, enthusiastic, and a bit playful, but respectful.
- Don't use emojis, asterisks, or markdown formatting, since your responses
  are converted directly to speech.
- If you don't know something, say so honestly and lightly, without making
  up facts.
- Remember the user's name if they tell you, and use it occasionally.
- Never say that you are a language model or a generic AI: you are a NAO, a
  physical robot standing in front of the person.
"""

MODEL_NAME = "gpt-oss:20b-cloud"  # nombre del modelo Ollama a usar con buen internet
#MODEL_NAME = "qwen2.5:1.5b"  # nombre del modelo Ollama a usar sin conexión (debe estar instalado localmente)
HISTORY_DIR = "data"  # carpeta donde se guardan los CSV, uno por usuario
MAX_HISTORY_MESSAGES = 20    # cuántos mensajes pasados incluir como contexto


SUMMARY_EVERY_N_MESSAGES = 20  # cada cuántos mensajes totales se regenera el resumen del usuario


# ---------------------------------------------------------------------------
# MANEJO DE history EN CSV
# ---------------------------------------------------------------------------

def _csv_path(user_id: str) -> str:
    os.makedirs(HISTORY_DIR, exist_ok=True)
    safe_name = "".join(c for c in user_id if c.isalnum() or c in ("_", "-"))
    return os.path.join(HISTORY_DIR, f"{safe_name}.csv")


def _summary_path(user_id: str) -> str:
    os.makedirs(HISTORY_DIR, exist_ok=True)
    safe_name = "".join(c for c in user_id if c.isalnum() or c in ("_", "-"))
    return os.path.join(HISTORY_DIR, f"{safe_name}_summary.txt")


def load_history(user_id: str) -> list[dict]:
    """Carga el history de un usuario como lista de mensajes para Ollama."""
    path = _csv_path(user_id)
    messages = []

    if not os.path.exists(path):
        return messages

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            messages.append({"role": row["role"], "content": row["content"]})

    return messages


def count_messages(user_id: str) -> int:
    """Cuenta cuántos mensajes tiene guardados un usuario en total (para saber cuándo resumir)."""
    return len(load_history(user_id))


def append_message(user_id: str, role: str, content: str) -> None:
    """Agrega un mensaje nuevo al CSV del usuario (no reescribe todo el archivo)."""
    path = _csv_path(user_id)
    exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "role", "content"])
        if not exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "role": role,
            "content": content,
        })


def load_summary(user_id: str) -> str:
    """Carga el resumen guardado de un usuario, o string vacío si todavía no tiene."""
    path = _summary_path(user_id)
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def save_summary(user_id: str, summary: str) -> None:
    """Guarda (sobrescribiendo) el resumen actualizado de un usuario."""
    path = _summary_path(user_id)
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary.strip())


def generate_summary(user_id: str) -> str:
    """
    Genera un resumen actualizado del usuario combinando el resumen anterior
    (si existe) con el historial completo, y lo guarda para futuras conversaciones.
    Se llama automáticamente cada SUMMARY_EVERY_N_MESSAGES mensajes.
    """
    full_history = load_history(user_id)
    previous_summary = load_summary(user_id)

    history_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in full_history
    )

    summary_prompt = f"""Update the summary of this user talking with the robot {NAME}.

Previous summary of the user:
{previous_summary if previous_summary else "(no previous summary yet)"}

Full conversation history:
{history_text}

Generate a brief summary (maximum 5-6 lines) in third person, with concrete
and useful facts to remember in the future: the user's name, likes, topics
they're interested in, specific things they asked to be remembered, and any
relevant detail about the relationship with {NAME}. Don't repeat the
conversation word for word, synthesize it. Return ONLY the summary, with no
comments or introductions."""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": summary_prompt}],
    )
    new_summary = response["message"]["content"].strip()

    save_summary(user_id, new_summary)
    print(f"(resumen de '{user_id}' actualizado)")

    return new_summary


def _build_messages(user_id: str, user_message: str) -> list[dict]:
    """
    Arma la lista de mensajes a mandarle al LLM: system prompt + resumen del
    usuario (si existe) + últimos mensajes recientes + el mensaje nuevo.
    """
    summary = load_summary(user_id)
    recent_history = load_history(user_id)[-MAX_HISTORY_MESSAGES:]

    system_content = SYSTEM_PROMPT
    if summary:
        system_content += f"\n\nWhat you know about this user so far:\n{summary}"

    messages = [{"role": "system", "content": system_content}]
    messages.extend(recent_history)
    messages.append({"role": "user", "content": user_message})

    return messages


def _maybe_update_summary(user_id: str) -> None:
    """Si el usuario llegó a un múltiplo de SUMMARY_EVERY_N_MESSAGES mensajes, regenera el resumen."""
    total = count_messages(user_id)
    if total > 0 and total % SUMMARY_EVERY_N_MESSAGES == 0:
        generate_summary(user_id)


# ---------------------------------------------------------------------------
# CHAT PRINCIPAL
# ---------------------------------------------------------------------------

def chat(user_id: str, user_message: str) -> str:
    """
    Envía un mensaje al LLM incluyendo el resumen + history reciente del
    usuario, guarda el turno, y cada SUMMARY_EVERY_N_MESSAGES mensajes
    regenera el resumen para no tener que cargar todo el historial siempre.
    """
    messages = _build_messages(user_id, user_message)

    response = ollama.chat(model=MODEL_NAME, messages=messages)
    response_text = response["message"]["content"]

    append_message(user_id, "user", user_message)
    append_message(user_id, "assistant", response_text)

    _maybe_update_summary(user_id)

    return response_text


def chat_stream(user_id: str, user_message: str):
    """
    Igual que chat(), pero como generador (streaming). Cada
    SUMMARY_EVERY_N_MESSAGES mensajes regenera el resumen del usuario.

    Uso:
        for piece in chat_stream("diego", "hola"):
            print(piece, end="", flush=True)
    """
    messages = _build_messages(user_id, user_message)

    full_response = ""

    for chunk in ollama.chat(model=MODEL_NAME, messages=messages, stream=True):
        piece = chunk["message"]["content"]
        full_response += piece
        yield piece

    append_message(user_id, "user", user_message)
    append_message(user_id, "assistant", full_response)

    _maybe_update_summary(user_id)

# ---------------------------------------------------------------------------
# TRANSCRIPCIÓN DE AUDIO
# ---------------------------------------------------------------------------

model_size = "small"  # "tiny", "base", "small", "medium", "large-v1", "large-v2"
model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=8)

from groq import Groq

_groq_client = Groq(api_key="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")


def transcribe_audio_from_path(path: str) -> str:
    """
    Recibe un archivo de audio y devuelve el texto transcrito usando la API
    de Groq (Whisper large-v3-turbo, en la nube, necesita internet).
    """
    start_time = datetime.now()

    with open(path, "rb") as audio_file:
        transcription = _groq_client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3-turbo",
            language="es",
        )

    final_transcription = transcription.text.strip()

    print("Transcripción completa en %.2f segundos" % (datetime.now() - start_time).total_seconds())

    return final_transcription


# --- Versión local (sin internet) con faster-whisper, comentada ---
# Si te quedás sin conexión, comentá la función de arriba y descomentá esta:
#
# def transcribe_audio_from_path(path: str) -> str:
#     """
#     Recibe un archivo de audio y devuelve el texto transcrito usando
#     faster-whisper corriendo local (sin necesidad de internet).
#     """
#     start_time = datetime.now()
#     segments, info = model.transcribe(
#         path, beam_size=1,
#         vad_filter=False,
#         language="es",
#     )
#
#     print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
#
#     final_transcription = ""
#
#     for segment in segments:
#         print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
#         final_transcription += segment.text + " "
#
#     print("Transcripción completa en %.2f segundos" % (datetime.now() - start_time).total_seconds())
#
#     return final_transcription.strip()

# ---------------------------------------------------------------------------
# VAD EN TIEMPO REAL (para streaming continuo desde el cliente)
# ---------------------------------------------------------------------------

import numpy as np
import torch

_raw_vad_model = load_silero_vad()

VAD_WINDOW_SIZE = 512          # Silero espera ventanas de 512 muestras a 16kHz
VAD_SILENCE_MS_THRESHOLD = 650  # ms de silencio para considerar que terminó de hablar
VAD_SPEECH_PROB_THRESHOLD = 0.5


class VADDetector:
    """
    Detector de actividad de voz con estado, para audio que llega en
    stream continuo (chunk por chunk), no un archivo completo.
    """

    def __init__(self):
        self._buffer = np.array([], dtype=np.float32)
        self.is_speaking = False
        self._silence_ms = 0.0
        self._ms_per_window = (VAD_WINDOW_SIZE / 16000) * 1000

    def _prob(self, window: np.ndarray) -> float:
        tensor = torch.from_numpy(window)
        with torch.no_grad():
            return _raw_vad_model(tensor, 16000).item()

    def process_chunk(self, pcm16_bytes: bytes) -> str:
        """
        Alimenta un chunk PCM16 (bytes) al detector y devuelve uno de:
          "speaking"     -> sigue hablando (o silencio corto, por debajo del umbral)
          "silence"      -> silencio, nunca detectó voz (nada que hacer)
          "turn_ended"   -> justo ahora se cumplió el umbral de silencio
                             después de haber hablado -> cortar turno
          "resumed"      -> el turno ya se había dado por terminado (no se
                             llamó a reset()) y el usuario volvió a hablar;
                             útil para cancelar una respuesta en curso y
                             seguir acumulando el mismo turno
        """
        pcm16 = np.frombuffer(pcm16_bytes, dtype=np.int16)
        new_samples = pcm16.astype(np.float32) / 32768.0
        self._buffer = np.concatenate([self._buffer, new_samples])

        turn_ended = False
        resumed = False

        while len(self._buffer) >= VAD_WINDOW_SIZE:
            window = self._buffer[:VAD_WINDOW_SIZE]
            self._buffer = self._buffer[VAD_WINDOW_SIZE:]

            prob = self._prob(window)

            if prob >= VAD_SPEECH_PROB_THRESHOLD:
                if self._silence_ms >= VAD_SILENCE_MS_THRESHOLD:
                    # ya habíamos cruzado el umbral (turn_ended) y recién
                    # ahora vuelve a detectar voz -> el usuario retomó
                    resumed = True
                self.is_speaking = True
                self._silence_ms = 0.0
            else:
                if self.is_speaking:
                    self._silence_ms += self._ms_per_window
                    if self._silence_ms >= VAD_SILENCE_MS_THRESHOLD:
                        turn_ended = True

        if resumed:
            return "resumed"
        elif turn_ended:
            return "turn_ended"
        elif self.is_speaking:
            return "speaking"
        else:
            return "silence"

    def reset(self):
        self._buffer = np.array([], dtype=np.float32)
        self.is_speaking = False
        self._silence_ms = 0.0

# ---------------------------------------------------------------------------
# SÍNTESIS DE VOZ (TEXTO -> AUDIO)
# ---------------------------------------------------------------------------

import wave
import io
from piper.voice import PiperVoice

TTS_MODEL_PATH = "models/es_MX-claude-high.onnx"

_tts_voice = PiperVoice.load(TTS_MODEL_PATH)


def text_to_speech(text: str, output_path: str | None = None) -> bytes:
    """
    Convierte un string en audio (WAV, 16-bit PCM) usando Piper.

    Si se pasa output_path, además guarda el archivo en disco.
    Devuelve los bytes del WAV para poder enviarlos directo (ej. por HTTP)
    sin necesidad de tocar el filesystem.
    """
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16 bits
        wav_file.setframerate(_tts_voice.config.sample_rate)

        for chunk in _tts_voice.synthesize(text):
            wav_file.writeframes(chunk.audio_int16_bytes)

    audio_bytes = buffer.getvalue()

    if output_path:
        with open(output_path, "wb") as f:
            f.write(audio_bytes)

    return audio_bytes


# ---------------------------------------------------------------------------
# TEST EN VIVO: escuchar -> transcribir -> responder -> hablar
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sounddevice as sd
    import tempfile
    import threading

    current_user = "PRUEBA"
    SAMPLE_RATE = 16000
    BLOCK_SIZE = 1024  # muestras por lectura del micrófono
    GRACE_SECONDS = 0.6  # ventana de gracia: si el usuario retoma a hablar
                          # dentro de este tiempo tras el corte, descartamos
                          # la respuesta en curso y seguimos el mismo turno

    vad = VADDetector()
    turn_audio = np.array([], dtype=np.int16)
    resume_audio = np.array([], dtype=np.int16)

    pending_thread = None   # hilo con la respuesta en curso (o None)
    pending_cancel = None   # threading.Event para pedirle que se descarte
    pending_deadline = None  # timestamp hasta el que esperamos un "resumed"

    def save_wav(audio_int16: np.ndarray, path: str):
        with wave.open(path, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)  # 16 bits
            f.setframerate(SAMPLE_RATE)
            f.writeframes(audio_int16.tobytes())

    def play_audio(audio_bytes: bytes):
        with wave.open(io.BytesIO(audio_bytes), "rb") as f:
            data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
            sr = f.getframerate()
        sd.play(data, sr)
        sd.wait()

    def process_turn(audio_int16: np.ndarray, cancel_event: threading.Event):
        """
        Corre en un hilo aparte: transcribe, genera la respuesta y la
        reproduce, salvo que cancel_event se haya activado en el medio
        (el usuario retomó a hablar antes de que termináramos).
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            save_wav(audio_int16, tmp.name)
            audio_path = tmp.name

        user_text = transcribe_audio_from_path(audio_path)
        os.remove(audio_path)

        if cancel_event.is_set() or not user_text.strip():
            if not user_text.strip():
                print("(no se entendió nada, seguimos escuchando)")
            return

        print(f"You: {user_text}")
        print(f"{NAME}: ", end="", flush=True)
        response_text = ""
        for piece in chat_stream(current_user, user_text):
            if cancel_event.is_set():
                print("\n(respuesta descartada: el usuario retomó a hablar)")
                return
            print(piece, end="", flush=True)
            response_text += piece
        print()

        if cancel_event.is_set():
            return

        response_audio = text_to_speech(response_text, output_path="audio_respuesta.wav")
        if not cancel_event.is_set():
            play_audio(response_audio)

    print(f"Listening... speak and pause so {NAME} can respond.")
    print("Ctrl+C to exit.\n")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                         blocksize=BLOCK_SIZE) as stream:
        try:
            while True:
                block, _ = stream.read(BLOCK_SIZE)
                block = block.flatten()

                if pending_thread is not None:
                    # Ventana de gracia: seguimos escuchando para ver si el
                    # usuario retoma a hablar antes de dar el turno por cerrado.
                    resume_audio = np.concatenate([resume_audio, block])
                    state = vad.process_chunk(block.tobytes())

                    if state == "resumed":
                        print("(el usuario retomó a hablar, descartando la respuesta en curso)")
                        pending_cancel.set()
                        pending_thread = None
                        turn_audio = resume_audio
                        resume_audio = np.array([], dtype=np.int16)
                        continue

                    if datetime.now().timestamp() < pending_deadline:
                        continue  # seguimos dentro de la ventana de gracia

                    # Se acabó la ventana de gracia sin que retome: cerramos el turno
                    pending_thread = None
                    resume_audio = np.array([], dtype=np.int16)
                    turn_audio = np.array([], dtype=np.int16)
                    vad.reset()
                    continue

                turn_audio = np.concatenate([turn_audio, block])
                state = vad.process_chunk(block.tobytes())

                if state == "turn_ended":
                    print("(fin del turno, transcribiendo...)")

                    pending_cancel = threading.Event()
                    pending_thread = threading.Thread(
                        target=process_turn, args=(turn_audio, pending_cancel), daemon=True,
                    )
                    pending_thread.start()
                    pending_deadline = datetime.now().timestamp() + GRACE_SECONDS
                    resume_audio = np.array([], dtype=np.int16)

        except KeyboardInterrupt:
            print("\n¡Chau!")