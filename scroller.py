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
            for id, lm in enumerate(hand_landmark.landmark):
                x = lm.x
                y = lm.y

    # Mostramos la imagen
    cv2.imshow("Hand Detection", image)

    # Método de cerrado
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberamos recursos
cap.release()
cv2.destroyAllWindows()
