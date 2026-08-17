# -*- coding: utf-8 -*-
from naoqi import ALProxy, ALModule, ALBroker
import websocket
import json
import threading

NAO_IP = "127.0.0.1"
NAO_PORT = 57642
SERVER_IP = "127.0.0.1"
SERVER_PORT = 8765
MY_IP = "127.0.0.1"
MY_PORT = 5000

is_sim = True

tts = ALProxy("ALTextToSpeech", NAO_IP, NAO_PORT)
aas = ALProxy("ALAnimatedSpeech", NAO_IP, NAO_PORT)
tts.setLanguage("Spanish")
tts.setParameter("speed", 150)
tts.setParameter("volume", 1.0)
aas.setBodyLanguageMode(1)  # 0: none, 1: context, 2: random

ws = websocket.create_connection("ws://%s:%d" % (SERVER_IP, SERVER_PORT))
ws_lock = threading.Lock()
print("Conectado al servidor")
user_id = "PRUEBA"


# ============================================================
# HILO QUE ESCUCHA RESPUESTAS DEL SERVIDOR, SIEMPRE ACTIVO
# ============================================================
def listener_loop():
    buffer = ""
    while True:
        try:
            respuesta = ws.recv()
        except Exception as e:
            print("Conexión cerrada del lado servidor:", e)
            break

        if not respuesta:
            continue

        try:
            data = json.loads(respuesta)
        except ValueError:
            continue

        tipo = data.get("type")

        if tipo == "response_chunk":
            buffer += data.get("text", "")
            if buffer and buffer[-1] in ".!?":
                print(buffer.strip())
                aas.say(buffer.strip().encode("utf-8"))
                buffer = ""

        elif tipo == "response_end":
            if buffer.strip():
                print(buffer.strip())
                aas.say(buffer.strip().encode("utf-8"))
            buffer = ""


listener_thread = threading.Thread(target=listener_loop)
listener_thread.daemon = True
listener_thread.start()


# ============================================================
# MÓDULO DE AUDIO (igual que antes, sin cambios)
# ============================================================
if is_sim:
    import sounddevice as sd
    import numpy as np
    SAMPLE_RATE = 16000
    CHANNELS = 1
    BLOCK_SIZE = 4096

    class FakeAudioModule:
        def __init__(self, name, websocket_conn, lock):
            self.name = name
            self.ws = websocket_conn
            self.lock = lock
            self.is_recording = False
            self._stream = None

        def _callback(self, indata, frames, time_info, status):
            if not self.is_recording:
                return
            if status:
                print(status)
            pcm16 = (indata[:, 0] * 32767).astype(np.int16)
            buffer_bytes = pcm16.tobytes()
            try:
                with self.lock:
                    self.ws.send(buffer_bytes, opcode=websocket.ABNF.OPCODE_BINARY)
            except Exception as e:
                print("Error mandando audio, deteniendo:", e)
                self.is_recording = False

        def start(self):
            self.is_recording = True
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32",
                blocksize=BLOCK_SIZE, callback=self._callback,
            )
            self._stream.start()
            print("Audio simulado iniciado (micrófono de la PC)")

        def stop(self):
            self.is_recording = False
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            print("Audio simulado detenido")

    FAM = FakeAudioModule("FakeAudioModule", ws, ws_lock)
else:
    broker = ALBroker("myBroker", MY_IP, 0, NAO_IP, NAO_PORT)
    AUD = ALProxy("ALAudioDevice", NAO_IP, NAO_PORT)

    class AudioModule(ALModule):
        def __init__(self, name, websocket_conn, lock):
            ALModule.__init__(self, name)
            self.audio = ALProxy("ALAudioDevice", NAO_IP, NAO_PORT)
            self.ws = websocket_conn
            self.lock = lock
            self.is_recording = False

        def start(self):
            self.audio.setClientPreferences(self.getName(), 16000, 3, 0)
            self.is_recording = True
            self.audio.subscribe(self.getName())
            print("Audio iniciado, escuchando...")

        def stop(self):
            self.is_recording = False
            self.audio.unsubscribe(self.getName())
            print("Audio detenido")

        def processRemote(self, nbOfChannels, nbrOfSamplesByChannel, timestamp, inputBuffer):
            if not self.is_recording:
                return
            try:
                with self.lock:
                    self.ws.send(inputBuffer, opcode=websocket.ABNF.OPCODE_BINARY)
            except Exception as e:
                print("Error mandando audio, deteniendo:", e)
                self.is_recording = False

    am = AudioModule("AudioModule", ws, ws_lock)


# ============================================================
# LOOP PRINCIPAL: solo manda mensajes, YA NO recibe (eso lo hace el listener)
# ============================================================
while True:
    mensaje = raw_input("Escribe un mensaje: ")

    if mensaje.lower() == "audio":
        print("Micrófono activado, streaming continuo...")
        if is_sim:
            FAM.start()
        else:
            am.start()
        continue

    if mensaje.lower() in ("salir", "exit"):
        break

    data = {"type": "chat", "user": user_id, "message": mensaje}
    with ws_lock:
        ws.send(json.dumps(data))
    # ya no hay ws.recv() aquí — el listener_thread se encarga de TODAS las respuestas

ws.close()