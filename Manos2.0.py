import cv2
import mediapipe as mp
import numpy as np

# Inicializamos MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Función para calcular distancia entre dos puntos
def calcular_distancia(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Inicializamos captura de video
cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
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

                # Obtenemos coordenadas de THUMB_TIP e INDEX_FINGER_TIP
                thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Convertimos a coordenadas de píxeles
                h, w, _ = frame.shape
                thumb_tip_px = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                index_tip_px = (int(index_tip.x * w), int(index_tip.y * h))

                # Dibujamos una línea entre los dos puntos
                cv2.line(frame, thumb_tip_px, index_tip_px, (0, 255, 0), 3)

                # Calculamos distancia y porcentaje (esto es arbitrario, ajusta según tu necesidad)
                distancia = calcular_distancia(thumb_tip_px, index_tip_px)
                porcentaje = np.clip((distancia / w) * 100, 0, 100)  # Porcentaje relativo al ancho de la pantalla

                # Mostramos el porcentaje en pantalla
                texto = f"Distancia: {distancia:.2f}px - {porcentaje:.2f}%"
                cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Mostramos el frame
        cv2.imshow('Detección de Mano', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Presiona ESC para salir
            break

cap.release()
cv2.destroyAllWindows()
