import cv2
import mediapipe as mp
import os
import math
import serial
import time

# Configurar puerto serial (ajustar el puerto según tu sistema)
ser = serial.Serial('COM8', 9600)
time.sleep(2)  # Esperar a que el puerto se inicialice

# Inicializar cámara
cap = cv2.VideoCapture(2)

# Configuración de MediaPipe Hands
mp_hands = mp.solutions.hands
manos = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Utilidad para dibujar
mp_dibujo = mp.solutions.drawing_utils

# Función para calcular distancia euclidiana entre dos puntos
def calcular_distancia(p1, p2):
    return math.sqrt((p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)

# Bucle principal
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = manos.process(frame_rgb)

    posiciones = []  # Almacena coordenadas de los landmarks

    if resultado.multi_hand_landmarks:
        for mano in resultado.multi_hand_landmarks:
            alto, ancho, _ = frame.shape

            for id, lm in enumerate(mano.landmark):
                corx, cory = int(lm.x * ancho), int(lm.y * alto)
                posiciones.append([id, corx, cory])

            # Dibujar la mano detectada
            mp_dibujo.draw_landmarks(frame, mano, mp_hands.HAND_CONNECTIONS)

            # Si detectamos los 21 puntos, verificamos si los dedos están doblados
            if len(posiciones) == 21:
                dedos = {"thumb": (4, 3), "index": (8, 6), "middle": (12, 10), "ring": (16, 14), "pinky": (20, 18)}
                dedos_doblados = []

                for nombre, (punta, base) in dedos.items():
                    if posiciones[punta][2] > posiciones[base][2]:  # Si la punta está más baja que la base
                        dedos_doblados.append(nombre)

                if dedos_doblados:
                    mensaje = f"Doblados: {', '.join(dedos_doblados)}\n"
                    ser.write(mensaje.encode())  # Enviar información por puerto serial
                    print(mensaje)

    # Mostrar el video
    cv2.imshow("Video", frame)

    # Salir con ESC
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
ser.close()