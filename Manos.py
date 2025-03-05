import cv2
import mediapipe as mp
import os
import math

# ----------------------------- Crear carpeta para almacenar fotos (si es necesario) ---------------------------------
nombre = 'Mano_Izquierda'
direccion = 'C:/Users/georl/OneDrive/Desktop/Trabajos_Universidad/SEPTIMO SEMESTRE 2025-I/PROYECTO INTEGRADOR 2025-I/Deteccion-y-Clasificacion-de-Manos-main/Fotos/Entrenamiento'
carpeta = direccion + '/' + nombre
if not os.path.exists(carpeta):
    print('Carpeta creada: ', carpeta)
    os.makedirs(carpeta)

cont = 0  # Contador de fotos

# Inicializar cámara
cap = cv2.VideoCapture(0)

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

            # Si detectamos los 21 puntos, calculamos distancias
            if len(posiciones) == 21:
                wrist = posiciones[0]
                thumb_tip = posiciones[4]
                index_tip = posiciones[8]
                middle_tip = posiciones[12]
                ring_tip = posiciones[16]
                pinky_tip = posiciones[20]

                # Calcular distancias desde la muñeca (wrist) a cada punta de dedo
                dist_thumb = calcular_distancia(wrist, thumb_tip)
                dist_index = calcular_distancia(wrist, index_tip)
                dist_middle = calcular_distancia(wrist, middle_tip)
                dist_ring = calcular_distancia(wrist, ring_tip)
                dist_pinky = calcular_distancia(wrist, pinky_tip)

                # Tomar la distancia máxima como referencia (dedo medio suele ser el más largo)
                max_dist = dist_middle

                # Calcular porcentajes
                percent_thumb = (dist_thumb / max_dist) * 100
                percent_index = (dist_index / max_dist) * 100
                percent_middle = (dist_middle / max_dist) * 100
                percent_ring = (dist_ring / max_dist) * 100
                percent_pinky = (dist_pinky / max_dist) * 100

                # Mostrar porcentajes sobre el video
                texto = f'Thumb: {percent_thumb:.1f}%, Index: {percent_index:.1f}%, Middle: {percent_middle:.1f}%'
                texto += f', Ring: {percent_ring:.1f}%, Pinky: {percent_pinky:.1f}%'

                cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # (Opcional) Mostrar las distancias reales en consola
                print(
                    f'Distancias (px) - Thumb: {dist_thumb:.1f}, Index: {dist_index:.1f}, Middle: {dist_middle:.1f}, Ring: {dist_ring:.1f}, Pinky: {dist_pinky:.1f}')

    # Mostrar el video
    cv2.imshow("Video", frame)

    # Salir con ESC o cuando se lleguen a 300 fotos (puedes quitar esto si solo quieres detección)
    k = cv2.waitKey(1)
    if k == 27 or cont >= 300:
        break

cap.release()
cv2.destroyAllWindows()