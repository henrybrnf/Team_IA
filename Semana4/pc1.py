import cv2
import numpy as np
import pygame
import sys

# ==============================
# CONFIGURACIÓN INICIAL
# ==============================
VIDEO_PATH = "video1.avi"  # usa 0 si quieres cámara web
MIN_CONTOUR_AREA = 6

# ------------------------------
# Inicializar captura de video
# ------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
if not ret:
    print("No se pudo abrir el video.")
    sys.exit(1)

alto, ancho = frame.shape[:2]

# ------------------------------
# Inicializar background subtractor
# ------------------------------
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)

# ------------------------------
# Configuración del Blob Detector
# ------------------------------
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 5
params.maxArea = 5000

params.filterByConvexity = True
params.minConvexity = 0.5

params.filterByInertia = True
params.minInertiaRatio = 0.1

if cv2.__version__.startswith("3.") or cv2.__version__.startswith("4."):
    detector = cv2.SimpleBlobDetector_create(params)
else:
    detector = cv2.SimpleBlobDetector(params)

# ------------------------------
# Inicializar pygame
# ------------------------------
pygame.init()
screen = pygame.display.set_mode((ancho, alto))
pygame.display.set_caption("Detección de burbujas")
font = pygame.font.SysFont("Arial", 16)
clock = pygame.time.Clock()

# Colores pygame
WHITE = (255, 255, 255)
BLUE = (0, 200, 255)
RED = (255, 0, 0)

frame_count = 0

# ==============================
# BUCLE PRINCIPAL
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin del video.")
        break

    frame_count += 1

    # --------------------------
    # Preprocesamiento OpenCV
    # --------------------------
    fgmask = fgbg.apply(frame)            # aplicar background subtraction
    fgmask_copy = fgmask.copy()           # copia de la máscara (lo pediste explícito)

    kernel = np.ones((3, 3), np.uint8)
    fgmask_clean = cv2.morphologyEx(fgmask_copy, cv2.MORPH_OPEN, kernel, iterations=1)
    fgmask_clean = cv2.morphologyEx(fgmask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Contornos
    contornos, hierarchy = cv2.findContours(fgmask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Detecciones
    for i, c in enumerate(contornos):
        area = cv2.contourArea(c)
        if area > MIN_CONTOUR_AREA:
            # ======= AQUÍ USAMOS cv2.boundingRect =======
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w // 2, y + h // 2

            # Dibujar rectángulo en OpenCV
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(frame, f"ID:{i}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(frame, f"({cx},{cy})", (x, y + h + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # --------------------------
    # Detección de blobs
    # --------------------------
    keypoints = detector.detect(fgmask_clean)
    im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]),
                                          (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # --------------------------
    # Mostrar en pygame
    # --------------------------
    im_rgb = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2RGB)
    surface = pygame.image.frombuffer(im_rgb.tobytes(), (ancho, alto), 'RGB')
    screen.blit(surface, (0, 0))

    # Líneas de referencia
    pygame.draw.line(screen, BLUE, (ancho // 2, 0), (ancho // 2, alto), 1)
    pygame.draw.line(screen, BLUE, (0, alto // 2), (ancho, alto // 2), 1)

    # Info
    text = font.render(f"Frame:{frame_count} Contornos:{len(contornos)} Keypoints:{len(keypoints)}",
                       True, WHITE)
    screen.blit(text, (10, 10))

    pygame.display.flip()
    clock.tick(30)

    # Eventos pygame
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            sys.exit(0)
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            cap.release()
            pygame.quit()
            sys.exit(0)

# ==============================
# FIN
# ==============================
cap.release()
pygame.quit()
cv2.destroyAllWindows()
print("Programa terminado correctamente.")
