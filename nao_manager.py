# -*- coding: utf-8 -*-
import time

from naoqi import ALProxy, ALModule, ALBroker
import websocket
import json
import threading

NAO_IP = "192.168.0.151"
NAO_PORT = 9559
SERVER_IP = "127.0.0.1"
SERVER_PORT = 8765
MY_IP = "127.0.0.1"
MY_PORT = 5000

is_sim = True

pm = ALProxy("PackageManager", NAO_IP, NAO_PORT)
uuid = pm.install("/home/nao/robot-language-spanish-1-2-1.pkg")
print("Instalado con uuid:", uuid)



motion = ALProxy("ALMotion", NAO_IP, NAO_PORT)
posture = ALProxy("ALRobotPosture", NAO_IP, NAO_PORT)
memory = ALProxy("ALMemory", NAO_IP, NAO_PORT)

motion.wakeUp()
posture.goToPosture("StandInit", 0.8)

tts = ALProxy("ALTextToSpeech", NAO_IP, NAO_PORT)
aas = ALProxy("ALAnimatedSpeech", NAO_IP, NAO_PORT)
tts.setLanguage("Spanish")
tts.setParameter("speed", 100)
aas.setBodyLanguageMode(1)  # 0: none, 1: context, 2: random

ws = websocket.create_connection("ws://%s:%d" % (SERVER_IP, SERVER_PORT))
ws_lock = threading.Lock()
print("Conectado al servidor")
user_id = "Crescencio"


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

        elif tipo == "response_end":
            if buffer.strip():
                print(buffer.strip())
                msg = {"type": "nao_speech_start"}
                ws.send(json.dumps(msg))
                print("NAO empieza a hablar...")
                aas.say(buffer.strip().encode("utf-8"))
            buffer = ""


listener_thread = threading.Thread(target=listener_loop)
listener_thread.daemon = True
listener_thread.start()


# ============================================================
# MÓDULO DE AUDIO 
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
            self.stream = None

        def callback(self, indata, frames, time_info, status):
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
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32",
                blocksize=BLOCK_SIZE, callback=self.callback,
            )
            self.stream.start()
            print("Audio simulado iniciado (micrófono de la PC)")

        def stop(self):
            self.is_recording = False
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
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
# HILO QUE MONITOREA SI EL NAO ESTÁ HABLANDO Y AVISA AL SERVIDOR
# ============================================================
def speech_status_loop():
    last_status = None
    while True:
        try:
            data = memory.getData("ALTextToSpeech/Status")
        except Exception as e:
            print("Error leyendo status TTS:", e)
            time.sleep(0.2)
            continue

        if data:
            _, status = data
            if status != last_status:
                print("Status TTS cambiado: " + str(last_status) + " -> " + (status))
                if status == "done":
                    msg = {"type": "nao_speech_end"}
                elif status == "starting":
                    msg = {"type": "nao_speech_start"}
                else:
                        msg = None

                if msg:
                    try:
                        with ws_lock:
                            ws.send(json.dumps(msg))
                    except Exception as e:
                        print("Error mandando status al server:", e)

                last_status = status

        time.sleep(0.1)


speech_thread = threading.Thread(target=speech_status_loop)
speech_thread.daemon = True
speech_thread.start()

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

ws.close()