import cv2
import numpy as np
image = cv2.imread('esferas.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray,10,150)
cv2.imshow('tono gris', gray)
#cv2.imshow('Canny',canny)
azul_bajo = np.array([110,50,50])
azul_alto = np.array([130,255,255])
hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
mascara_azul = cv2.inRange(hsv,azul_bajo,azul_alto)
mascara_visualizado = cv2.bitwise_and(image,image,mask=mascara_azul)
cv2.imshow('mascarasoloazul',mascara_visualizado)

cv2.waitKey(0)           # Espera hasta que presiones una tecla
cv2.destroyAllWindows()  # Cierra todas las ventanas abiertas
