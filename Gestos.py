import cv2
import mediapipe as mp
import numpy as np
import serial
import time

# Inicializa la comunicación serial (ajusta el puerto y la velocidad según tu configuración)
ser = serial.Serial('COM4', 9600, timeout=1)  # Cambia 'COM4' por el puerto correcto
time.sleep(2)  # Espera a que el puerto esté listo

# Inicializamos MediaPipe Hands y utilidades para dibujar
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def contar_dedos(landmarks, frame_shape, handedness="Right"):
    h, w, _ = frame_shape

    # Coordenadas para el pulgar
    thumb_tip = (int(landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * w),
                 int(landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * h))
    thumb_ip = (int(landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * w),
                int(landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * h))

    # Coordenadas para el índice
    index_tip = (int(landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w),
                 int(landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h))
    index_pip = (int(landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * w),
                 int(landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * h))

    # Coordenadas para el dedo medio
    middle_tip = (int(landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * w),
                  int(landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * h))
    middle_pip = (int(landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * w),
                  int(landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * h))

    # Coordenadas para el anular
    ring_tip = (int(landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * w),
                int(landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * h))
    ring_pip = (int(landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * w),
                int(landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * h))

    # Coordenadas para el meñique
    pinky_tip = (int(landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * w),
                 int(landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * h))
    pinky_pip = (int(landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * w),
                 int(landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * h))

    dedos = 0

    # Lógica para el pulgar: se utiliza la clasificación de la mano.
    # Para mano derecha, se considera levantado si la punta está a la derecha del IP.
    if handedness == "Right":
        if thumb_tip[0] > thumb_ip[0]:
            dedos += 1
    else:  # Para mano izquierda se invierte la comparación
        if thumb_tip[0] < thumb_ip[0]:
            dedos += 1

    # Para el resto de los dedos se considera levantado si la punta está más arriba (valor y menor) que el PIP.
    if index_tip[1] < index_pip[1]:
        dedos += 1
    if middle_tip[1] < middle_pip[1]:
        dedos += 1
    if ring_tip[1] < ring_pip[1]:
        dedos += 1
    if pinky_tip[1] < pinky_pip[1]:
        dedos += 1

    return dedos


# Inicializamos la captura de video
cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = hands.process(rgb_frame)

        if resultado.multi_hand_landmarks:
            # Obtenemos la clasificación de la mano (Right o Left)
            handedness = "Right"
            if resultado.multi_handedness:
                handedness = resultado.multi_handedness[0].classification[0].label

            for landmarks in resultado.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                dedos_contados = contar_dedos(landmarks, frame.shape, handedness)
                cv2.putText(frame, f"Dedo(s) levantado(s): {dedos_contados}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Se envía el número de dedos contados al Arduino
                ser.write(f"{dedos_contados}\n".encode('utf-8'))

        cv2.imshow("Conteo de Dedos", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Presiona ESC para salir
            break

cap.release()
cv2.destroyAllWindows()
ser.close()
