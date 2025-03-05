import cv2
import mediapipe as mp
import numpy as np
import serial
import time

# Inicializa la comunicación serial (ajusta COM según tu PC y velocidad según el Arduino)
ser = serial.Serial('COM3', 9600, timeout=1)  # Cambia 'COM3' según corresponda
time.sleep(2)  # Espera que el puerto esté listo

# Inicializamos MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Función para calcular distancia entre dos puntos
def calcular_distancia(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Inicializamos captura de video
cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                    min_detection_confidence=0.5, 
                    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertimos a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesamos con MediaPipe
        resultado = hands.process(rgb_frame)

        if resultado.multi_hand_landmarks:
            for landmarks in resultado.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                # Obtenemos coordenadas de puntos clave
                h, w, _ = frame.shape
                thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
                pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                # Convertimos a coordenadas de píxeles
                thumb_tip_px = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                index_tip_px = (int(index_tip.x * w), int(index_tip.y * h))
                wrist_px = (int(wrist.x * w), int(wrist.y * h))
                pinky_tip_px = (int(pinky_tip.x * w), int(pinky_tip.y * h))

                # Dibujamos línea entre pulgar e índice
                cv2.line(frame, thumb_tip_px, index_tip_px, (0, 255, 0), 3)

                # Distancia entre pulgar e índice (gesto)
                dist_gesto = calcular_distancia(thumb_tip_px, index_tip_px)

                # Distancia relativa de la mano (distancia entre muñeca y meñique)
                dist_mano = calcular_distancia(wrist_px, pinky_tip_px)

                # Calcular porcentaje ajustado por distancia a la cámara
                porcentaje_base = np.clip((dist_gesto / w) * 100, 0, 100)
                
                # Factor de escala basado en el tamaño de la mano
                factor_distancia = np.clip(1 - (dist_mano / w), 0.5, 1)  # Más lejos -> más pequeño
                porcentaje_ajustado = int(porcentaje_base * factor_distancia)

                # Mostrar en pantalla
                cv2.putText(frame, f"Distancia: {dist_gesto:.2f}px", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Porcentaje: {porcentaje_ajustado}%", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Enviar el valor ajustado al Arduino
                ser.write(f"{porcentaje_ajustado}\n".encode('utf-8'))

        # Mostrar el frame
        cv2.imshow('Detección de Mano', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
            break

cap.release()
cv2.destroyAllWindows()
ser.close()
