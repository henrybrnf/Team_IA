import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2

class IslandFinderApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Analizador de Huellas Dactilares - Detección de Isla/Punto (v3 Mejorado)")
        self.master.geometry("1800x950") # Ampliamos un poco la ventana para el nuevo panel

        self.original_pil_image = None
        self.display_size = (350, 350)
        
        # --- PARÁMETROS CLAVE DEL ALGORITMO ---
        # Ahora que limpiamos la imagen, podemos ser más estrictos.
        # Una isla real tendrá muy pocos píxeles de longitud.
        self.MIN_ISLAND_PIXELS = 1  # Longitud mínima en píxeles
        self.MAX_ISLAND_PIXELS = 10 # Longitud máxima. Reducido para mayor precisión.

        # --- Configuración de la Interfaz ---
        title_label = tk.Label(self.master, text="Proceso de Detección de Minucia 'Isla' o 'Punto' con Limpieza Morfológica", font=("Helvetica", 16, "bold"), pady=10)
        title_label.pack(side=tk.TOP)
        
        control_frame = tk.Frame(self.master, pady=5)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.btn_load = tk.Button(control_frame, text="Cargar Imagen", command=self.load_image, font=("Helvetica", 10))
        self.btn_load.pack(side=tk.LEFT, padx=20)
        
        self.btn_process = tk.Button(control_frame, text="Detectar Islas/Puntos", command=self.process_image, state=tk.DISABLED, font=("Helvetica", 10))
        self.btn_process.pack(side=tk.LEFT, padx=10)

        # Reconfiguramos la grilla para 2x3 paneles
        self.grid_frame = tk.Frame(self.master, padx=10, pady=10)
        self.grid_frame.pack(fill=tk.BOTH, expand=True)
        for i in range(2): self.grid_frame.rowconfigure(i, weight=1)
        for i in range(3): self.grid_frame.columnconfigure(i, weight=1)
        
        self.lbl_original = self._create_image_panel("1. Imagen Original", 0, 0)
        self.lbl_binarized = self._create_image_panel("2. Imagen Binarizada", 0, 1)
        self.lbl_cleaned = self._create_image_panel("3. Imagen Limpia (Morfológica)", 0, 2)
        self.lbl_thinned = self._create_image_panel("4. Adelgazamiento (Esqueleto)", 1, 0)
        self.lbl_final = self._create_image_panel("5. Resultado (Islas/Puntos)", 1, 1)
        
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
        if not file_path: return
        self.original_pil_image = Image.open(file_path)
        self._update_image_label(self.lbl_original, self.original_pil_image)
        for label in [self.lbl_binarized, self.lbl_cleaned, self.lbl_thinned, self.lbl_final]:
            label.config(image="", text="Esperando análisis...")
            label.image = None
        self.lbl_results.config(text="Listo para analizar")
        self.btn_process.config(state=tk.NORMAL)

    def process_image(self):
        if self.original_pil_image is None:
            messagebox.showerror("Error", "Primero debes cargar una imagen.")
            return
        
        try:
            # 1. Binarización
            img_gray = self.original_pil_image.convert("L")
            img_np = np.array(img_gray)
            _, binarized_np = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            self._update_image_label(self.lbl_binarized, binarized_np)
            self.master.update_idletasks()
            
            # 2. **NUEVO PASO: Limpieza Morfológica**
            kernel = np.ones((3,3), np.uint8)
            # La apertura (MORPH_OPEN) elimina el ruido tipo "sal" (puntos blancos aislados)
            cleaned_np = cv2.morphologyEx(binarized_np, cv2.MORPH_OPEN, kernel, iterations=1)
            self._update_image_label(self.lbl_cleaned, cleaned_np)
            self.master.update_idletasks()

            # 3. Adelgazamiento (ahora sobre la imagen limpia)
            thinned_np = cv2.ximgproc.thinning(cleaned_np, thinningType=cv2.ximgproc.THINNING_GUOHALL)
            self._update_image_label(self.lbl_thinned, thinned_np)
            self.master.update_idletasks()
            
            # 4. Extracción de Minucias (Isla/Punto)
            island_centers, island_count = self.detect_islands(thinned_np)
            
            # 5. Dibujo de resultados
            img_para_dibujar = cv2.cvtColor(thinned_np, cv2.COLOR_GRAY2RGB)
            
            for (x, y) in island_centers:
                size = 5
                p1 = (x, y - size)
                p2 = (x - size, y + size // 2)
                p3 = (x + size, y + size // 2)
                triangle_cnt = np.array([p1, p2, p3], dtype=np.int32)
                cv2.drawContours(img_para_dibujar, [triangle_cnt], 0, (0, 0, 255), 1)

            self._update_image_label(self.lbl_final, img_para_dibujar)
            self.lbl_results.config(text=f"ANÁLISIS COMPLETO   |   Islas/Puntos encontrados: {island_count}", font=("Helvetica", 12, "bold"))
        except Exception as e:
            messagebox.showerror("Error de Procesamiento", f"Ocurrió un error inesperado: {e}")
            self.lbl_results.config(text="Análisis fallido.", font=("Helvetica", 12, "italic"))

    def detect_islands(self, thinned_image):
        """
        Detecta minucias de tipo Isla/Punto utilizando el análisis de componentes conectados
        sobre una imagen previamente limpiada.
        """
        island_centers = []
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thinned_image, 8, cv2.CV_32S)
        
        for i in range(1, num_labels):
            pixel_count = stats[i, cv2.CC_STAT_AREA]
            if self.MIN_ISLAND_PIXELS <= pixel_count <= self.MAX_ISLAND_PIXELS:
                cx, cy = centroids[i]
                island_centers.append((int(cx), int(cy)))
                
        return island_centers, len(island_centers)

if __name__ == "__main__":
    root = tk.Tk()
    app = IslandFinderApp(root)
    root.mainloop()