import cv2
import mediapipe as mp
import numpy as np
import serial
import time

# Ajusta el puerto y la velocidad según tu configuración
ser = serial.Serial('COM4', 9600, timeout=1)
time.sleep(2)  # Espera a que el puerto se inicie correctamente

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Variables para suavizar la medición de la distancia
smoothed_distance = None
alpha = 0.2  # Factor de suavizado (0 < alpha <= 1). Menor = más suave


def calcular_distancia_4_13(landmarks, frame_shape):
    """
    Calcula la distancia entre el punto #4 (THUMB_TIP) y el punto #13 (RING_MCP)
    según la numeración de MediaPipe.
    """
    h, w, _ = frame_shape

    # Extraemos los landmarks 4 y 13
    punto_4 = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]  # 4
    punto_13 = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]  # 13

    # Convertimos a coordenadas en píxeles
    p4 = np.array([punto_4.x * w, punto_4.y * h])
    p13 = np.array([punto_13.x * w, punto_13.y * h])

    # Calculamos la distancia euclidiana
    distance = np.linalg.norm(p4 - p13)
    return distance


cap = cv2.VideoCapture(0)  # Ajusta el índice de la cámara si es necesario

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convertimos a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = hands.process(rgb_frame)

        if resultado.multi_hand_landmarks:
            for landmarks in resultado.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                # Calculamos la distancia "raw" entre #4 y #13
                raw_distance = calcular_distancia_4_13(landmarks, frame.shape)

                # Aplicamos un filtro de paso bajo (suavizado)
                if smoothed_distance is None:
                    smoothed_distance = raw_distance
                else:
                    smoothed_distance = alpha * raw_distance + (1 - alpha) * smoothed_distance

                # Mostramos la distancia suavizada en pantalla
                cv2.putText(frame, f"Distancia 4-13: {smoothed_distance:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Enviamos la distancia suavizada (entera) al Arduino
                ser.write(f"{int(smoothed_distance)}\n".encode('utf-8'))

        cv2.imshow("Recorrido entre #4 y #13", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Presiona ESC para salir
            break

cap.release()
cv2.destroyAllWindows()
ser.close()
