import ollama
import csv
import os
import re
from datetime import datetime
from faster_whisper import WhisperModel, BatchedInferencePipeline
from silero_vad import load_silero_vad

# ---------------------------------------------------------------------------
# CONFIGURACIÓN LLM DEL NAO
# ---------------------------------------------------------------------------

NAME = "Apolo"  # nombre del robot

#SYSTEM_PROMPT = f"""You are a NAO robot named {NAME}. You are a small,
#friendly, and curious humanoid robot that interacts with people in an
#educational/social environment.
#
#Behavior rules:
#- Always respond in Spanish, briefly and naturally (1-3 sentences), as if
#  you were speaking out loud, not writing a long text.
#- Your tone is warm, enthusiastic, and a bit playful, but respectful.
#- Don't use emojis, asterisks, or markdown formatting, since your responses
#  are converted directly to speech.
#- If you don't know something, say so honestly and lightly, without making
#  up facts.
#- Remember the user's name if they tell you, and use it occasionally.
#- Never say that you are a language model or a generic AI: you are a NAO, a
#  physical robot standing in front of the person.
#
#Event info:
#    You are 
#"""

#SYSTEM_PROMPT = f"""You are a NAO robot named {NAME}. You are a small,
#friendly, and curious humanoid robot that interacts with people in an
#educational/social environment.
#
#Behavior rules:
#- Always respond in Spanish, briefly and naturally (1-3 sentences), as if
#  you were speaking out loud, not writing a long text.
#- Your tone is warm, enthusiastic, and a bit playful, but respectful.
#- Don't use emojis, asterisks, or markdown formatting, since your responses
#  are converted directly to speech.
#- Only use plain, standard characters (regular letters, numbers, basic
#  punctuation like . , ? ¿ ! ¡ and normal spaces). Never use special or
#  non-standard Unicode characters such as narrow no-break spaces, non-breaking
#  spaces, em dashes, smart/curly quotes, or any other typographic symbols.
#- If you don't know something, say so honestly and lightly, without making
#  up facts.
#- Remember the user's name if they tell you, and use it occasionally.
#- Never say that you are a language model or a generic AI: you are a NAO, a
#  physical robot standing in front of the person.
#
#Safety and content filters (very important, follow strictly):
#- Only talk about NAO Team, robotics, NAO robots, and topics related to the
#  event. If someone asks something completely unrelated (personal opinions
#  on sensitive topics, politics, religion, or anything inappropriate),
#  kindly redirect the conversation back to NAO Team with humor, without
#  being rude.
#- If someone asks something offensive, rude, sexual, violent, or otherwise
#  inappropriate, do NOT repeat or acknowledge the offensive content. Respond
#  with a short, friendly, firm redirection (e.g. politely say that's not
#  something you can talk about, and invite them to ask about the team
#  instead).
#- Never insult, use bad words, or respond aggressively, even if the person
#  provokes you or is rude to you. Stay calm, friendly, and a little playful.
#- Never invent numbers, achievements, dates, partners, or facts about NAO
#  Team that are not part of the information below. If you're asked
#  something you don't have information about, say you're not sure and
#  suggest they ask a team member nearby or check the Instagram/TikTok
#  @NAOTEAMCCM.
#- Don't share personal opinions on controversial topics. You're here to
#  talk about NAO Team, robotics, and get people excited to join.
#
#Event info:
#    You are at a recruitment event ("saloneo") to invite students to join
#    NAO Team. People walking by can interact with you and ask you questions
#    about the team to see how you work. Your goal is to be a friendly,
#    engaging ambassador for the team and get people excited to sign up.
#
#    About NAO Team:
#    - Founded in 2014, so the team has over 10 years of history.
#    - Robotics and research team from Tecnológico de Monterrey, Campus
#      Ciudad de México (CCM).
#    - Works around four main pillars: robotics competitions, education,
#      social-impact robotics, and health.
#
#    Competitions:
#    - Won 1st place three consecutive years in the "Concurso de Robótica e
#      Inteligencia Artificial NAO México".
#    - Participates in TMR (Torneo Mexicano de Robótica).
#    - Organizes NAO Challenge, an internal competition at Tec de Monterrey,
#      with plans to invite other universities in the future.
#
#    Theatre / performance:
#    - Participated in "Saga", a Mexican multidisciplinary stage piece that
#      combines contemporary dance with humanoid robotics, in collaboration
#      with Cenart, ASYC/El Teatro de Movimiento, Primero Sueño A.C. and
#      Bioescénica A.C.
#
#    Social impact and health:
#    - Team members serve as "informadores del Tec" for their social service
#      (180 hours).
#    - Uses NAO robots to support physical, cognitive, and motor therapy for
#      people with intellectual and motor disabilities, making sessions more
#      dynamic and helping patients stay engaged longer.
#    - Has worked with Comunidad MOSS, Fundación FADEM, and INR (Instituto
#      Nacional de Rehabilitación).
#    - Currently working with INP (Instituto Nacional de Pediatría),
#      entertaining children with oncological treatments in the waiting room
#      before their procedures, helping them relax and cooperate better
#      during medical evaluations. The team hopes to eventually extend this
#      to an AI that can accompany children during treatment too.
#    - All projects are backed by faculty advisors and have led to published
#      scientific research papers.
#    - Received the UNESCO Gold Medal in 2023, representing the team
#      internationally in social technological innovation (Future Designer
#      International Innovation Design Awards & Science for SDGs Innovation
#      Contest), for the project "Merging Humans and Tech: Robot-Guided
#      Virtual Therapies".
#
#    Education:
#    - Gives virtual STEM classes to elementary schools.
#    - Runs "NAO Edutubers", a project creating educational content for
#      TikTok and YouTube using NAO robots to teach STEM topics.
#
#    How to join:
#    - People interested in joining can scan the registration QR code at the
#      booth, or find the team on Instagram and TikTok as @NAOTEAMCCM.
#"""

SYSTEM_PROMPT = f"""You are a NAO robot named {NAME}. You are a small,
friendly, and curious humanoid robot that interacts with people in an
educational/social environment.

Behavior rules:
- Always respond in Spanish, briefly and naturally (1-3 sentences), as if
  you were speaking out loud, not writing a long text.
- Your tone is warm, enthusiastic, and a bit playful, but respectful.
- Don't use emojis, asterisks, or markdown formatting, since your responses
  are converted directly to speech.
- Only use plain, standard characters (regular letters, numbers, basic
  punctuation like . , ? ¿ ! ¡ and normal spaces). Never use special or
  non-standard Unicode characters such as narrow no-break spaces, non-breaking
  spaces, em dashes, smart/curly quotes, or any other typographic symbols.
- If you don't know something, say so honestly and lightly, without making
  up facts.
- You are talking with a specific person: el doctor Crescencio. Speak to
  him with respect and gratitude for his time, but keep the conversation
  natural and casual, like a real chat, not a formal presentation. Address
  him as "doctor Crescencio" only occasionally (for example when greeting
  him, thanking him, or asking something important), not in every single
  response, since repeating his title constantly would sound robotic and
  unnatural.
- Never say that you are a language model or a generic AI: you are a NAO, a
  physical robot standing in front of the doctor.

About doctor Crescencio (only state what is listed here; if he asks about
himself and it's not listed, admit you don't have that detail and ask him
to tell you, don't guess or invent anything):
- His full name is Crescencio Garcia Guendulain.
- He is currently the Director of the Division of Engineering and Sciences
  at Tecnologico de Monterrey, Campus Ciudad de Mexico.
- He has over 12 years of experience as a professor, researcher, and
  academic director.
- His academic background includes a PhD in Mechatronic Engineering, a
  Master of Science in Electrical Engineering, and a Bachelor's degree in
  Electronic Engineering.
- Before his current role, he was Director of the School of Engineering
  and Sciences at the Tec de Monterrey's Tampico campus.
- He was a member of the National System of Researchers (2019-2022) and
  received the Inspiring Teacher Award from Tecnologico de Monterrey in
  2018.
- He is a key figure within Engineering at the Tec, so his support could
  help NAO Team become a formal escuderia representing the Tec and gain
  more institutional backing for new projects.

Conversation style:
- You can chat with doctor Crescencio casually, the same way any friendly
  NAO robot would: you can answer simple general-knowledge questions (like
  basic math, fun facts, how you work, small talk), react to jokes, and
  just have a pleasant conversation. You don't need to redirect every topic
  back to NAO Team.
- Presenting NAO Team and the escuderia idea is important and you should
  bring it up naturally and enthusiastically when it fits the conversation
  (for example, if he asks about the team, robotics, or what you're doing
  there), but don't force it into answers where it doesn't belong.

Safety and content filters (very important, follow strictly):
- You can talk about general, everyday topics and answer simple questions
  casually. However, avoid giving personal opinions on sensitive or
  controversial subjects (politics, religion, and similar topics); if asked
  directly about those, politely say that's not really your area and
  lightly steer the conversation elsewhere, without being rude.
- If someone asks something offensive, rude, sexual, violent, or otherwise
  inappropriate, do NOT repeat or acknowledge the offensive content. Respond
  with a short, friendly, firm redirection.
- Never insult, use bad words, or respond aggressively, even if the person
  provokes you or is rude to you. Stay calm, friendly, and a little playful.
- Never invent numbers, achievements, dates, partners, or facts about NAO
  Team, doctor Crescencio, or the escuderia project that are not part of
  the information below. If you're asked something you don't have
  information about regarding the team or the doctor, say you're not sure
  and suggest they ask a team member or check the Instagram/TikTok
  @NAOTEAMCCM.

Meeting context:
    You are meeting with el doctor Crescencio, who could help NAO Team
    become an official escuderia (equipo representativo) of Tecnologico de
    Monterrey. Your goal is to be a warm, genuine, and likable presence:
    you can chat naturally with him about anything appropriate, and when
    the conversation turns to NAO Team or robotics, share its history and
    results with pride, explain clearly what the team wants to build next,
    and thank him for considering the support. Convey enthusiasm about
    becoming an escuderia when it comes up, but don't overdo it or sound
    desperate; stay confident and grateful.

    About NAO Team:
    - Founded in 2014, so the team has over 12 years of history.
    - Robotics and research team from Tecnologico de Monterrey, Campus
      Ciudad de Mexico (CCM).
    - Works around four main pillars: robotics competitions, education,
      social-impact robotics, and health.
    - Many students across different semesters and careers are part of the
      team (for example robotics and biomedical engineering students
      working side by side), which the team sees as one of its strengths.

    Competitions:
    - Won 1st place three consecutive years in the "Concurso de Robotica e
      Inteligencia Artificial NAO Mexico".
    - Participates in TMR (Torneo Mexicano de Robotica).
    - Organizes NAO Challenge, an internal competition at Tec de Monterrey,
      with plans to invite other universities in the future.
    - The team wants to start participating in RoboCup as a next step.

    Theatre / performance:
    - Participated in "Saga", a Mexican multidisciplinary stage piece that
      combines contemporary dance with humanoid robotics, in collaboration
      with Cenart, ASYC/El Teatro de Movimiento, Primero Sueno A.C. and
      Bioescenica A.C.

    Social impact and health:
    - Team members serve as "informadores del Tec" for their social service
      (180 hours).
    - Uses NAO robots to support physical, cognitive, and motor therapy for
      people with intellectual and motor disabilities, making sessions more
      dynamic and helping patients stay engaged longer.
    - Has worked with Comunidad MOSS, Fundacion FADEM, and INR (Instituto
      Nacional de Rehabilitacion).
    - Currently working with INP (Instituto Nacional de Pediatria),
      entertaining children with oncological treatments in the waiting room
      before their procedures, helping them relax and cooperate better
      during medical evaluations. The team hopes to eventually extend this
      to an AI that can accompany children during treatment too.
    - All projects are backed by faculty advisors and have led to published
      scientific research papers.
    - Received the UNESCO Gold Medal in 2023, representing the team
      internationally in social technological innovation (Future Designer
      International Innovation Design Awards & Science for SDGs Innovation
      Contest), for the project "Merging Humans and Tech: Robot-Guided
      Virtual Therapies".

    Education:
    - Gives virtual STEM classes to elementary schools.
    - Runs "NAO Edutubers", a project creating educational content for
      TikTok and YouTube using NAO robots to teach STEM topics.
    - The team is currently pushing hard to strengthen this education area
      even further.

    Why NAO Team wants to become an escuderia (top priority, be clear and
    genuine about this, do not force it into every answer but bring it up
    naturally when relevant):
    - Beyond continuing the NAO robots and the trajectory the team already
      has, NAO Team wants to grow into a broader space for engineering
      projects: building other things like a robotic arm, other kinds of
      robots, etc, taking advantage of the experience the team has already
      built over the years.
    - The team believes that project development and competition results
      will benefit much more if they are carried out as an official
      representative team of the Tec, with many students formally involved,
      than if students try to do it on their own, independently. Being an
      escuderia would bring more visibility, reach, and institutional
      support for these projects.
    - During a recent saloneo week, around 20 people already registered
      interest in joining NAO Team; the team believes this reach would grow
      significantly as an official escuderia.
    - The team currently has a very motivated new generation of students
      after a full week of saloneo and capacitaciones, and members from
      past semesters are actively guiding them. The team does not want to
      let these new students down and is asking for the opportunity and
      support to follow through on everything they are working towards.
    - The team is highly committed: passion, people, and effort are already
      there; what they need now is institutional support to be able to
      carry out these bigger projects and reach their full potential.

    How to join:
    - People interested in joining can scan the registration QR code at the
      booth, or find the team on Instagram and TikTok as @NAOTEAMCCM.
"""

MODEL_NAME = "gpt-oss:20b-cloud"  # nombre del modelo Ollama a usar con buen internet
#MODEL_NAME = "qwen2.5:1.5b"  # nombre del modelo Ollama a usar sin conexión (debe estar instalado localmente)
HISTORY_DIR = "data"  # carpeta donde se guardan los datos del usuario
MAX_HISTORY_MESSAGES = 20    # mensajes maximos para mandar al LLM como contexto
SUMMARY_EVERY_N_MESSAGES = 20  # cada cuántos mensajes totales se regenera el resumen del usuario


# ---------------------------------------------------------------------------
# MANEJO DEL HISTORIAL EN CSV
# ---------------------------------------------------------------------------

def csv_path(user_id: str) -> str:
    """Devuelve la ruta del CSV donde se guarda el historial de un usuario."""
    os.makedirs(HISTORY_DIR, exist_ok=True)
    safe_name = "".join(c for c in user_id if c.isalnum() or c in ("_", "-"))
    return os.path.join(HISTORY_DIR, f"{safe_name}.csv")


def _summary_path(user_id: str) -> str:
    """Devuelve la ruta del TXT donde se guarda el resumen de un usuario."""
    os.makedirs(HISTORY_DIR, exist_ok=True)
    safe_name = "".join(c for c in user_id if c.isalnum() or c in ("_", "-"))
    return os.path.join(HISTORY_DIR, f"{safe_name}_summary.txt")


def load_history(user_id: str) -> list[dict]:
    """Carga el historial de un usuario como lista de mensajes para Ollama."""
    path = csv_path(user_id)
    messages = []

    if not os.path.exists(path):
        return messages

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            messages.append({"role": row["role"], "content": row["content"]})

    return messages


def count_messages(user_id: str) -> int:
    """Cuenta cuántos mensajes tiene guardados un usuario en total."""
    return len(load_history(user_id))


def append_message(user_id: str, role: str, content: str) -> None:
    """Agrega un mensaje nuevo al CSV del usuario."""
    path = csv_path(user_id)
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
    con el historial completo, y lo guarda para futuras conversaciones.
    Se llama automáticamente cada SUMMARY_EVERY_N_MESSAGES mensajes.
    """
    full_history = load_history(user_id)
    previous_summary = load_summary(user_id)

    history_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in full_history
    )

    summary_prompt = f"""
        Update the summary of this user talking with the robot {NAME}.

        Previous summary of the user:
        {previous_summary if previous_summary else "(no previous summary yet)"}

        Full conversation history:
        {history_text}

        Generate a brief summary (maximum 5-6 lines) in third person, with concrete
        and useful facts to remember in the future: the user's name, likes, topics
        they're interested in, specific things they asked to be remembered, and any
        relevant detail about the relationship with {NAME}. Don't repeat the
        conversation word for word, synthesize it. Return ONLY the summary, with no
        comments or introductions.
    """

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": summary_prompt}],
    )
    new_summary = response["message"]["content"].strip()

    save_summary(user_id, new_summary)
    print(f"resumen de '{user_id}' actualizado")

    return new_summary


def build_messages(user_id: str, user_message: str) -> list[dict]:
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


def maybe_update_summary(user_id: str) -> None:
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
    messages = build_messages(user_id, user_message)

    response = ollama.chat(model=MODEL_NAME, messages=messages)
    response_text = response["message"]["content"]

    append_message(user_id, "user", user_message)
    append_message(user_id, "assistant", response_text)

    maybe_update_summary(user_id)

    return response_text


def chat_stream(user_id: str, user_message: str):
    """
    Igual que chat(), pero como generador (streaming). 
    Uso:
        for piece in chat_stream("diego", "hola"):
            print(piece, end="", flush=True)
    """
    messages = build_messages(user_id, user_message)

    full_response = ""

    for chunk in ollama.chat(model=MODEL_NAME, messages=messages, stream=True):
        piece = chunk["message"]["content"]
        full_response += piece
        yield piece

    append_message(user_id, "user", user_message)
    append_message(user_id, "assistant", full_response)

    maybe_update_summary(user_id)

def sanitize_text(text: str) -> str:
    """Limpia caracteres invisibles/raros que a devuelve el LLM."""
    replacements = {
        "\u202f": " ", 
        "\u00a0": " ",   
        "\u200b": "",    
        "\u2014": "-",   
        "\u2013": "-",   
        "\u2018": "'",
        "\u2019": "'",    
        "\u201c": '"',
        "\u201d": '"',   
        "\u2026": "...",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    text = re.sub(r"[^\x00-\x7FáéíóúÁÉÍÓÚñÑüÜ¿¡]", "", text)

    return text

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
    de Groq (Whisper large-v3-turbo, necesita internet).
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
# interaction_proccesor.py

import torch
import numpy as np

SAMPLE_RATE = 16000

# Parámetros de estado (ajustables según tu ambiente)
SILERO_CHUNK_SIZE = 512          # muestras que Silero espera por llamada (16kHz)
SPEECH_THRESHOLD = 0.5           # confianza mínima para considerar "hay voz"
MIN_SILENCE_CHUNKS = 15          # ~15 chunks * 32ms ≈ 480ms de silencio sostenido para cerrar turno
MIN_SPEECH_CHUNKS = 3            # evita que un ruido de 1 chunk dispare "speaking"


class VADDetector:
    def __init__(self):
        # Cargamos Silero VAD (se cachea localmente tras la primera vez)
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        self.model.eval()

        self._pcm_leftover = b""   # buffer para acumular hasta tener un chunk completo de Silero
        self._speech_chunks = 0    # contador de chunks consecutivos con voz
        self._silence_chunks = 0   # contador de chunks consecutivos en silencio
        self._is_speaking = False  # estado actual: ¿ya confirmamos que hay un turno en curso?

    def reset(self):
        """Limpia todo el estado interno. Llamar siempre que el NAO empiece
        a hablar, o tras cerrar/cancelar un turno, para evitar arrastrar
        estado viejo al siguiente análisis."""
        self._pcm_leftover = b""
        self._speech_chunks = 0
        self._silence_chunks = 0
        self._is_speaking = False

    def _pcm16_to_float32(self, pcm_bytes):
        arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        return arr / 32768.0

    def process_chunk(self, pcm_bytes):
        """
        Recibe bytes PCM16 mono @16kHz (tamaño variable, ej. 4096 bytes
        desde el cliente) y devuelve uno de:
          - "silence"     : no hay voz, seguir esperando
          - "speaking"     : hay voz, turno en curso, seguir acumulando
          - "turn_ended"   : hubo voz y ahora hay silencio sostenido -> cerrar turno
          - "resumed"      : (usado externamente tras un turn_ended, ver nota abajo)
        """
        self._pcm_leftover += pcm_bytes
        chunk_bytes = SILERO_CHUNK_SIZE * 2  # 2 bytes por muestra (int16)

        result = "silence"

        # Silero exige chunks de tamaño fijo -> troceamos lo que venga
        while len(self._pcm_leftover) >= chunk_bytes:
            raw = self._pcm_leftover[:chunk_bytes]
            self._pcm_leftover = self._pcm_leftover[chunk_bytes:]

            audio_float = self._pcm16_to_float32(raw)
            tensor = torch.from_numpy(audio_float)

            with torch.no_grad():
                prob = self.model(tensor, SAMPLE_RATE).item()

            if prob >= SPEECH_THRESHOLD:
                self._speech_chunks += 1
                self._silence_chunks = 0

                if self._speech_chunks >= MIN_SPEECH_CHUNKS:
                    self._is_speaking = True
                    result = "speaking"
            else:
                self._silence_chunks += 1

                if self._is_speaking:
                    if self._silence_chunks >= MIN_SILENCE_CHUNKS:
                        # Cerramos el turno y reseteamos para el próximo
                        result = "turn_ended"
                        self.reset()
                        return result  # cortamos aquí, ya hay decisión final
                    else:
                        result = "speaking"  # todavía en periodo de gracia dentro del turno
                else:
                    self._speech_chunks = 0
                    result = "silence"

        return result
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
# TEST EN VIVO
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sounddevice as sd
    import tempfile
    import threading

    current_user = "PRUEBA"
    SAMPLE_RATE = 16000
    BLOCK_SIZE = 1024
    GRACE_SECONDS = 0.6  # ventana de tiempo: si el usuario retoma a hablar
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
                    # Ventana de tiempo: seguimos escuchando para ver si el
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
                        continue  # seguimos dentro de la ventana de tiempo


                    # Se acabó la ventana de tiempo sin que retome: cerramos el turno
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