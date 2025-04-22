import cv2
import mediapipe as mp

# Importo las soluciones para las manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Inicializamos la cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Error al leer un frame")
        break

    # Convertimos la imagen a RGB (formato compatible con MediaPipe)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Procesamos la imagen
    results = hands.process(img_rgb)

    # Dibujamos las marcas si hay manos detectadas
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmark, mp_hands.HAND_CONNECTIONS)

            # deteccion de dedos
            indice = hand_landmark.landmark[8].y < hand_landmark.landmark[6].y
            medio = hand_landmark.landmark[12].y < hand_landmark.landmark[10].y
            anular = hand_landmark.landmark[16].y > hand_landmark.landmark[14].y
            menique = hand_landmark.landmark[20].y > hand_landmark.landmark[18].y
            pulgar = hand_landmark.landmark[4].x < hand_landmark.landmark[3].x

            if all([indice, medio, anular, menique, pulgar]):
                print("bajar")
            elif indice and not all([medio, anular, menique, pulgar]):
                print("subir")
            elif not all([indice, medio, anular, menique, pulgar]):
                print("detenerse")
            else:
                print("otra accion")

        # Mostramos la imagen
    cv2.imshow("Hand Detection", image)

    # Método de cerrado
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberamos recursos
cap.release()
cv2.destroyAllWindows()
