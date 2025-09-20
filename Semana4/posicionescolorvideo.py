import cv2
import numpy as np
# import serial
# arduino = serial.Serial('COM3', 9600)

cap = cv2.VideoCapture('esferamovimiento.mp4')

while cap.isOpened():
    ret, image = cap.read()
    if not ret:  # Si no se pudo leer el frame (fin de video o error)
        print("Fin del video o error al leer frame.")
        break

    m, n, _ = image.shape
    cv2.line(image, (0, int(m/2)), (n, int(m/2)), (0, 0, 255), 1)
    cv2.line(image, (int(n/2), 0), (int(n/2), m), (0, 0, 255), 1)

    naranja_bajo = np.array([1, 190, 20])
    naranja_alto = np.array([18, 255, 255])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mascara_naranja = cv2.inRange(hsv, naranja_bajo, naranja_alto)

    cnts, _ = cv2.findContours(mascara_naranja, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) > 10:
            x, y, w, h = cv2.boundingRect(approx)
            mensaje = f"({x},{y})"
            if y < m/2:
                if x < n/2:
                    print("Arriba-Izquierda")
                    mensaje = "II " + mensaje
                    # arduino.write(b'i')
                else:
                    print("Arriba-Derecha")
                    mensaje = "I " + mensaje
                    # arduino.write(b'd')
            else:
                if x < n/2:
                    print("Abajo-Izquierda")
                    mensaje = "III " + mensaje
                    # arduino.write(b'i')
                else:
                    print("Abajo-Derecha")
                    mensaje = "IV " + mensaje
                    # arduino.write(b'd')

            cv2.putText(image, mensaje, (x, y-5), 1, 1, (255, 255, 255), 1)
            break

    cv2.imshow('imagen', image)
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # Tecla ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
# arduino.close()
