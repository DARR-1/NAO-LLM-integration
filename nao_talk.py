# -*- coding: utf-8 -*-
"""
nao_animated_say.py

Script para NAO (NAOqi SDK) que:
  1. Se conecta al robot.
  2. Espera que el usuario ingrese un texto por consola (el "input").
  3. Hace que NAO diga ese texto usando ALAnimatedSpeech (habla con gestos).

Requisitos:
  - Python 2.7 (NAOqi SDK clásico usa Python 2.7, no Python 3).
  - Tener instalado el "pynaoqi" / NAOqi Python SDK correspondiente a la
    versión de tu robot (descargable desde el Aldebaran/SoftBank Robotics
    developer portal).
  - Saber la IP y el puerto del robot (por defecto el puerto es 9559).
    La IP la puedes ver apretando el botón del pecho del NAO (la dice
    por voz) o revisando el router.

Uso:
  python nao_animated_say.py --ip 192.168.1.50 --port 9559
"""

import sys
import argparse

try:
    from naoqi import ALProxy
except ImportError:
    print("ERROR: No se encontro el modulo 'naoqi'. Asegurate de tener el "
          "NAOqi SDK instalado y de estar corriendo este script con Python 2.7, "
          "con el SDK agregado al PYTHONPATH.")
    sys.exit(1)


def conectar_robot(ip, puerto):
    """Crea los proxies necesarios para hablar con el robot."""
    try:
        tts_animado = ALProxy("ALAnimatedSpeech", ip, puerto)
    except Exception as e:
        print("No se pudo crear el proxy a ALAnimatedSpeech. Revisa IP/puerto.")
        print("Detalle: {}".format(e))
        sys.exit(1)
    return tts_animado


def decir_texto_animado(tts_animado, texto):
    """
    Hace que el robot diga el texto usando animated say.
    Se pueden agregar tags de animacion manualmente, por ejemplo:
        "^start(animations/Stand/Gestures/Explain_1) Hola ^wait(...)"
    Pero por defecto, ALAnimatedSpeech ya agrega gestos automaticos
    basados en el contenido del texto (modalidad "contextual").
    """
    configuracion = {"bodyLanguageMode": "contextual"}
    tts_animado.say(texto, configuracion)


def loop_consola(tts_animado):
    """Loop principal: espera input por consola y hace que NAO lo diga."""
    print("Conectado al robot. Escribe un texto y presiona Enter para que "
          "NAO lo diga (o escribe 'salir' para terminar).")
    while True:
        texto = raw_input("Texto para NAO > ")  # raw_input es de Python 2
        if texto.strip().lower() in ("salir", "exit", "quit"):
            print("Terminando programa.")
            break
        if texto.strip() == "":
            continue
        decir_texto_animado(tts_animado, texto)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control de NAO: animated say")
    parser.add_argument("--ip", type=str, default="192.168.0.151",
                         help="IP del robot NAO")
    parser.add_argument("--port", type=int, default=9559,
                         help="Puerto de NAOqi (por defecto 9559)")
    args = parser.parse_args()

    tts_animado = conectar_robot(args.ip, args.port)
    loop_consola(tts_animado)