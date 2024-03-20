import cv2
import mediapipe as mp
import time
import os
import re
import pygame


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Inicializar pygame para reproducir audio
pygame.mixer.init()
countdown_sound = pygame.mixer.Sound('countdown_sound.mp3')  # Cargar el sonido para el conteo

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Variable para controlar la visibilidad del texto en la pantalla
show_countdown = False


# Variables para controlar la reproducción del sonido del contador
last_second = -1
sound_played = False

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # 1 para detectar solo una mano
    min_detection_confidence=0.5) as hands:

    capturing_image = False  # Bandera para indicar si estamos capturando una imagen
    capture_start_time = None
    capture_interval = 5  # Intervalo de tiempo entre capturas (en segundos)
    image_counter = 1  # Contador para asignar nombres únicos a las imágenes
    reset_timer = True  # Bandera para reiniciar el temporizador
    show_message = False  # Bandera para mostrar el mensaje de "Imagen capturada"

    # Obtener el número más alto en los nombres de archivo existentes
    existing_images = [filename for filename in os.listdir() if re.match(r'captured_image_(\d+)\.jpg', filename)]
    if existing_images:
        image_numbers = [int(re.match(r'captured_image_(\d+)\.jpg', filename).group(1)) for filename in existing_images]
        image_counter = max(image_numbers) + 1

    while True:
        ret, frame = cap.read()
        if ret == False:
            break
            
        # Tamaño de la pantalla
        screen_width = 1280  # Ancho de la pantalla
        screen_height = 720  # Alto de la pantalla

        # Redimensionar el fotograma al tamaño de la pantalla
        frame = cv2.resize(frame, (screen_width, screen_height))

        # Mostrar el fotograma redimensionado
        cv2.imshow('Frame', frame)

        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        # Dibujar temporizador en la ventana
        elapsed_time = 0
        if capturing_image or reset_timer:
            elapsed_time = time.time() - capture_start_time if capturing_image else 0
            remaining_time = max(0, capture_interval - elapsed_time)
            show_countdown = capturing_image

            
            current_second = int(remaining_time)  # Obtener el segundo actual

            # Reproducir sonido si el segundo actual es diferente al segundo previamente reproducido
            if current_second != last_second:
                last_second = current_second
                sound_played = False  # Restablecer el indicador de sonido

            # Emitir sonido cada segundo si aún no se ha reproducido para ese segundo
            if capturing_image and current_second > 0 and not sound_played:
                countdown_sound.play()  # Reproducir el sonido del conteo
                sound_played = True  # Actualizar el indicador de sonido


            
        if elapsed_time >= capture_interval + 1:
            while True:
                if f'captured_image_{image_counter}.jpg' not in existing_images:
                    cv2.imwrite(f'captured_image_{image_counter}.jpg', frame)
                    print(f'Imagen {image_counter} capturada')

                    # Reproducir sonido al capturar la imagen
                    pygame.mixer.music.load('camera_shutter.mp3')
                    pygame.mixer.music.play()

                    capturing_image = False  # Detener la captura después de guardar la imagen
                    image_counter += 1  # Incrementar el contador para el nombre de la siguiente imagen
                    reset_timer = True  # Reiniciar el temporizador
                    show_message = True  # Mostrar mensaje "Imagen capturada"
                    start_message_time = time.time()  # Marcar el tiempo de inicio para mostrar el mensaje
                    break
                else:
                    image_counter += 1


        if show_message:
            text = 'Imagen capturada'
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 4, 6)[0]
            text_x = int((screen_width - text_size[0]) / 2)  # Posición x para centrar el texto horizontalmente
            text_y = int((screen_height + text_size[1]) / 2)  # Posición y para centrar el texto verticalmente
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 8, cv2.LINE_AA)
            if time.time() - start_message_time > 1:  # Mostrar el mensaje por 1 segundo
                show_message = False

        if results.multi_hand_landmarks is not None and len(results.multi_hand_landmarks) == 1:
            hand_landmarks = results.multi_hand_landmarks[0]

            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_finger = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Verificar si se hace el gesto de "bien"
            index_middle_distance = ((index_finger.x - middle_finger.x)**2 + (index_finger.y - middle_finger.y)**2)**0.5
            other_fingers_closed = ring_finger.y < hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y and \
                                    pinky.y < hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y and \
                                    thumb.y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y

            # Verificar si la palma está hacia adelante
            palm_facing_forward = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z


            if thumb.y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y and \
               thumb.x > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x and \
               thumb.x < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x and \
               thumb.y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y and \
               thumb.y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y and \
               thumb.y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y:
                   
                if not capturing_image:
                    capturing_image = True
                    capture_start_time = time.time()


            if palm_facing_forward:
                reset_timer = True  # Reiniciar el temporizador si se detecta la palma hacia adelante
                capturing_image = False  # Detener la captura si la palma está hacia adelante

        if results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
        # Mostrar el frame con o sin texto del temporizador dependiendo de la variable show_countdown
        if show_countdown:
            text = str(int(remaining_time) + 1)  # Mostrar números enteros y sumar 1 para compensar el retraso
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 10, 10)[0]
            text_x = int((screen_width - text_size[0]) / 2)
            text_y = int((screen_height + text_size[1]) / 2)

            # Conteo regresivo en el centro de la pantalla
            cv2.putText(frame, text, (text_x, text_y), font, 10, (255, 255, 255), 20, cv2.LINE_AA)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()