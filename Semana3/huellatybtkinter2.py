import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2

class FingerprintApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Analizador de Huellas Dactilares - Optimizado")
        self.master.geometry("1600x900")

        self.original_pil_image = None
        self.display_size = (350, 350) 

        # --- Configuración de la Interfaz ---
        title_label = tk.Label(self.master, text="Proceso de Análisis de Huella Dactilar", font=("Helvetica", 16, "bold"), pady=10)
        title_label.pack(side=tk.TOP)
        control_frame = tk.Frame(self.master, pady=5)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        self.btn_load = tk.Button(control_frame, text="Cargar Imagen", command=self.load_image, font=("Helvetica", 10))
        self.btn_load.pack(side=tk.LEFT, padx=20)
        self.btn_process = tk.Button(control_frame, text="Analizar Huella", command=self.process_image, state=tk.DISABLED, font=("Helvetica", 10))
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
        self.lbl_final = self._create_image_panel("4. Resultado (Minucias Marcadas)", 1, 1)
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
        
        # --- 1. Conversión y Preparación ---
        img_gray = self.original_pil_image.convert("L")
        img_np = np.array(img_gray)

        # --- 2. Binarización (Método de Otsu) ---
        # Se usa THRESH_BINARY_INV para que las crestas queden en blanco (255)
        # y los valles en negro (0), que es lo estándar para el adelgazamiento.
        _, binarized_np = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self._update_image_label(self.lbl_binarized, binarized_np)
        self.master.update_idletasks()
        
        # --- 3. Adelgazamiento (Usando OpenCV - Rápido y Eficiente) ---
        thinned_np = cv2.ximgproc.thinning(binarized_np)
        self._update_image_label(self.lbl_thinned, thinned_np)
        self.master.update_idletasks()
        
        # --- 4. Extracción y Validación de Minucias ---
        # Convertimos la imagen adelgazada (0s y 255s) a booleana (False/True)
        # para que sea compatible con la función de análisis.
        thinned_bool = thinned_np > 0
        img_remarcada, terminaciones, bifurcaciones = self.analizar_y_remarcar(thinned_bool)
        self._update_image_label(self.lbl_final, img_remarcada)
        
        # --- 5. Mostrar Resultados ---
        self.lbl_results.config(text=f"ANÁLISIS COMPLETO  |  Terminaciones: {terminaciones}  |  Bifurcaciones: {bifurcaciones}", font=("Helvetica", 12, "bold"))

    def analizar_y_remarcar(self, Img_bool):
        # La función ahora recibe la imagen adelgazada y booleana.
        m, n = Img_bool.shape
        T = np.zeros((m, n))
        B = np.zeros((m, n))

        # --- Extracción de minucias con Crossing Number ---
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                if Img_bool[i, j]:
                    # Tomamos la región de 3x3 y la convertimos a entero (0 o 1)
                    R = Img_bool[i-1:i+2, j-1:j+2].astype(int)
                    P1,P2,P3 = R[0,0], R[0,1], R[0,2]
                    P8, _ ,P4 = R[1,0], R[1,1], R[1,2]
                    P7,P6,P5 = R[2,0], R[2,1], R[2,2]
                    
                    # Cálculo del Crossing Number
                    CN = 0.5 * (abs(P2-P1) + abs(P3-P2) + abs(P4-P3) + abs(P5-P4) + abs(P6-P5) + abs(P7-P6) + abs(P8-P7) + abs(P1-P8))
                    
                    if CN == 1.0: T[i, j] = 1 # Terminación
                    if CN == 3.0: B[i, j] = 1 # Bifurcación

        # --- Validación/Filtro de minucias ---
        # Elimina minucias que están agrupadas en un vecindario de 3x3
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                if T[i,j]: T[i,j] = not (T[i-1:i+2,j-1:j+2].sum() > 1)
                if B[i,j]: B[i,j] = not (B[i-1:i+2,j-1:j+2].sum() > 1)
        
        terminacion, bifurcacion = 0, 0
        
        # --- Crear imagen final y dibujar minucias ---
        esqueleto_uint8 = (Img_bool * 255).astype(np.uint8)
        img_para_dibujar = cv2.cvtColor(esqueleto_uint8, cv2.COLOR_GRAY2RGB)
        
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                if T[i, j]:
                    terminacion += 1
                    cv2.rectangle(img_para_dibujar, (j-3, i-3), (j+3, i+3), (255, 0, 0), 1) # Rojo para Terminaciones
                if B[i, j]:
                    bifurcacion += 1
                    cv2.circle(img_para_dibujar, (j, i), 4, (0, 0, 255), 1) # Azul para Bifurcaciones
        
        return img_para_dibujar, terminacion, bifurcacion

if __name__ == "__main__":
    root = tk.Tk()
    app = FingerprintApp(root)
    root.mainloop()