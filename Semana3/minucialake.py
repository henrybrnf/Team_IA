import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2

class LakeFinderApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Analizador de Huellas Dactilares - Detección de minucia Lake")
        self.master.geometry("1600x900")

        self.original_pil_image = None
        self.display_size = (350, 350) 
        
        # --- PARÁMETROS CLAVE DEL ALGORITMO DE LAGOS ---
        # Estos valores definen qué tan grande o pequeño puede ser un "agujero" para ser considerado un lago.
        # Puedes ajustarlos para diferentes calidades de imagen.
        self.MIN_LAKE_AREA = 5    # Área mínima en píxeles
        self.MAX_LAKE_AREA = 150  # Área máxima en píxeles

        # --- Configuración de la Interfaz ---
        title_label = tk.Label(self.master, text="Proceso de Detección de Minucia 'Lake'", font=("Helvetica", 16, "bold"), pady=10)
        title_label.pack(side=tk.TOP)
        
        control_frame = tk.Frame(self.master, pady=5)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.btn_load = tk.Button(control_frame, text="Cargar Imagen", command=self.load_image, font=("Helvetica", 10))
        self.btn_load.pack(side=tk.LEFT, padx=20)
        
        self.btn_process = tk.Button(control_frame, text="Detectar Lakes", command=self.process_image, state=tk.DISABLED, font=("Helvetica", 10))
        self.btn_process.pack(side=tk.LEFT, padx=10)

        self.grid_frame = tk.Frame(self.master, padx=10, pady=10)
        self.grid_frame.pack(fill=tk.BOTH, expand=True)
        self.grid_frame.rowconfigure(0, weight=1)
        self.grid_frame.rowconfigure(1, weight=1)
        self.grid_frame.columnconfigure(0, weight=1)
        self.grid_frame.columnconfigure(1, weight=1)
        
        self.lbl_original = self._create_image_panel("1. Imagen Original", 0, 0)
        self.lbl_binarized = self._create_image_panel("2. Imagen Binarizada (Otsu)", 0, 1)
        self.lbl_thinned = self._create_image_panel("3. Adelgazamiento (Esqueleto)", 1, 0)
        self.lbl_final = self._create_image_panel("4. Resultado (Lakes Marcados)", 1, 1)
        
        self.lbl_results = tk.Label(self.master, text="Resultados aparecerán aquí", font=("Helvetica", 12, "italic"), pady=10)
        self.lbl_results.pack(side=tk.BOTTOM)

    def _create_image_panel(self, title, row, col):
        frame = tk.LabelFrame(self.grid_frame, text=title, font=("Helvetica", 11, "bold"), padx=10, pady=10)
        frame.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)
        label = tk.Label(frame)
        label.pack(fill=tk.BOTH, expand=True)
        return label

    def _update_image_label(self, label, image_data):
        if isinstance(image_data, np.ndarray):
            pil_image = Image.fromarray(image_data)
        else:
            pil_image = image_data
        
        pil_image = pil_image.resize(self.display_size, Image.Resampling.LANCZOS)
        photo_image = ImageTk.PhotoImage(pil_image)
        label.config(image=photo_image, text="")
        label.image = photo_image
        
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.tif *.jpg *.png *.bmp")])
        if not file_path:
            return
        self.original_pil_image = Image.open(file_path)
        self._update_image_label(self.lbl_original, self.original_pil_image)
        for label in [self.lbl_binarized, self.lbl_thinned, self.lbl_final]:
            label.config(image="", text="Esperando análisis...")
            label.image = None
        self.lbl_results.config(text="Listo para analizar")
        self.btn_process.config(state=tk.NORMAL)

    def process_image(self):
        if self.original_pil_image is None:
            messagebox.showerror("Error", "Primero debes cargar una imagen.")
            return
        
        # 1. Binarización
        img_gray = self.original_pil_image.convert("L")
        img_np = np.array(img_gray)
        _, binarized_np = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self._update_image_label(self.lbl_binarized, binarized_np)
        self.master.update_idletasks()
        
        # 2. Adelgazamiento
        thinned_np = cv2.ximgproc.thinning(binarized_np)
        self._update_image_label(self.lbl_thinned, thinned_np)
        self.master.update_idletasks()
        
        # 3. Extracción de Minucias (con validación interna)
        lake_centers, lake_count = self.detect_lakes(thinned_np)
        
        # 4. Dibujo de resultados
        img_para_dibujar = cv2.cvtColor(thinned_np, cv2.COLOR_GRAY2RGB)
        
        for (x, y) in lake_centers:
            # Dibuja un cuadrado verde con grosor de línea de 1 píxel
            cv2.rectangle(img_para_dibujar, (x-4, y-4), (x+4, y+4), (0, 255, 0), 1)

        self._update_image_label(self.lbl_final, img_para_dibujar)
        self.lbl_results.config(text=f"ANÁLISIS COMPLETO   |   Lakes encontrados: {lake_count}", font=("Helvetica", 12, "bold"))

    def detect_lakes(self, thinned_image):
        """
        Detecta minucias de tipo lago y las valida por tamaño.
        Devuelve una lista de las coordenadas (x, y) del centro de cada lago validado.
        """
        lake_centers = []
        image_copy = thinned_image.copy()

        # Extracción: Encuentra todos los contornos y su jerarquía
        contours, hierarchy = cv2.findContours(image_copy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        if hierarchy is None:
            return [], 0

        # Validación: Itera y filtra los contornos que son "agujeros" y cumplen el criterio de área
        for i in range(len(contours)):
            if hierarchy[0][i][3] != -1: # Es un agujero (tiene un contorno padre)
                contour = contours[i]
                area = cv2.contourArea(contour)
                
                if self.MIN_LAKE_AREA < area < self.MAX_LAKE_AREA: # Criterio de validación
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        lake_centers.append((cX, cY))
        
        return lake_centers, len(lake_centers)


if __name__ == "__main__":
    root = tk.Tk()
    app = LakeFinderApp(root)
    root.mainloop()