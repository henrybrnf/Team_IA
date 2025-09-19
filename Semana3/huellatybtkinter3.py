import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2

class FingerprintApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Analizador de Huellas Dactilares - Con Filtros")
        self.master.geometry("1600x900")

        self.original_pil_image = None
        self.display_size = (350, 350) 

        # NUEVO: Variables para guardar los resultados del análisis
        self.thinned_image_bool = None
        self.terminations_map = None
        self.bifurcations_map = None

        # --- Configuración de la Interfaz ---
        title_label = tk.Label(self.master, text="Proceso de Análisis de Huella Dactilar", font=("Helvetica", 16, "bold"), pady=10)
        title_label.pack(side=tk.TOP)
        
        control_frame = tk.Frame(self.master, pady=5)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.btn_load = tk.Button(control_frame, text="Cargar Imagen", command=self.load_image, font=("Helvetica", 10))
        self.btn_load.pack(side=tk.LEFT, padx=20)
        
        self.btn_process = tk.Button(control_frame, text="Analizar Huella", command=self.process_image, state=tk.DISABLED, font=("Helvetica", 10))
        self.btn_process.pack(side=tk.LEFT, padx=10)

        # NUEVO: Variables de control para los Checkbuttons
        self.show_terminations = tk.BooleanVar(value=True)
        self.show_bifurcations = tk.BooleanVar(value=True)

        # NUEVO: Creación de los Checkbuttons
        self.cb_terminations = tk.Checkbutton(control_frame, text="Mostrar Terminaciones", variable=self.show_terminations, command=self._redraw_final_image, state=tk.DISABLED, font=("Helvetica", 10))
        self.cb_terminations.pack(side=tk.LEFT, padx=10)
        
        self.cb_bifurcations = tk.Checkbutton(control_frame, text="Mostrar Bifurcaciones", variable=self.show_bifurcations, command=self._redraw_final_image, state=tk.DISABLED, font=("Helvetica", 10))
        self.cb_bifurcations.pack(side=tk.LEFT, padx=5)

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
        # NUEVO: Desactivar y resetear checkbuttons al cargar nueva imagen
        self.cb_terminations.config(state=tk.DISABLED)
        self.cb_bifurcations.config(state=tk.DISABLED)
        self.show_terminations.set(True)
        self.show_bifurcations.set(True)


    def process_image(self):
        if self.original_pil_image is None:
            messagebox.showerror("Error", "Primero debes cargar una imagen.")
            return
        
        img_gray = self.original_pil_image.convert("L")
        img_np = np.array(img_gray)

        _, binarized_np = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self._update_image_label(self.lbl_binarized, binarized_np)
        self.master.update_idletasks()
        
        thinned_np = cv2.ximgproc.thinning(binarized_np)
        self._update_image_label(self.lbl_thinned, thinned_np)
        self.master.update_idletasks()
        
        # Guardamos los resultados para no tener que recalcularlos
        self.thinned_image_bool = thinned_np > 0
        
        # La función de análisis ahora solo calcula y devuelve los mapas de minucias
        self.terminations_map, self.bifurcations_map = self.analizar_minucias(self.thinned_image_bool)
        
        # La primera vez, dibujamos la imagen final
        self._redraw_final_image()
        
        # NUEVO: Activamos los checkbuttons ahora que el análisis está completo
        self.cb_terminations.config(state=tk.NORMAL)
        self.cb_bifurcations.config(state=tk.NORMAL)

    # NUEVO: Función para redibujar la imagen final según los checkbuttons
    def _redraw_final_image(self):
        if self.thinned_image_bool is None:
            return # No hacer nada si no hay una imagen procesada

        # Recuperar el estado de los checkbuttons
        show_term = self.show_terminations.get()
        show_bif = self.show_bifurcations.get()

        # Crear la imagen base (el esqueleto)
        esqueleto_uint8 = (self.thinned_image_bool * 255).astype(np.uint8)
        img_para_dibujar = cv2.cvtColor(esqueleto_uint8, cv2.COLOR_GRAY2RGB)

        terminacion_count = 0
        bifurcacion_count = 0
        
        m, n = self.thinned_image_bool.shape

        # Dibujar solo las minucias seleccionadas
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                if self.terminations_map[i, j]:
                    terminacion_count +=1
                    if show_term:
                        cv2.rectangle(img_para_dibujar, (j-3, i-3), (j+3, i+3), (255, 0, 0), 1) # Rojo
                
                if self.bifurcations_map[i, j]:
                    bifurcacion_count += 1
                    if show_bif:
                        cv2.circle(img_para_dibujar, (j, i), 4, (0, 0, 255), 1) # Azul

        # Actualizar el panel de la imagen final
        self._update_image_label(self.lbl_final, img_para_dibujar)
        # Actualizar el texto de resultados
        self.lbl_results.config(text=f"ANÁLISIS COMPLETO  |  Terminaciones encontradas: {terminacion_count}  |  Bifurcaciones encontradas: {bifurcacion_count}", font=("Helvetica", 12, "bold"))

    # MODIFICADO: La función ahora solo analiza y devuelve los mapas
    def analizar_minucias(self, Img_bool):
        m, n = Img_bool.shape
        T = np.zeros((m, n))
        B = np.zeros((m, n))

        for i in range(1, m - 1):
            for j in range(1, n - 1):
                if Img_bool[i, j]:
                    R = Img_bool[i-1:i+2, j-1:j+2].astype(int)
                    P1,P2,P3 = R[0,0], R[0,1], R[0,2]
                    P8, _ ,P4 = R[1,0], R[1,1], R[1,2]
                    P7,P6,P5 = R[2,0], R[2,1], R[2,2]
                    
                    CN = 0.5 * (abs(P2-P1) + abs(P3-P2) + abs(P4-P3) + abs(P5-P4) + abs(P6-P5) + abs(P7-P6) + abs(P8-P7) + abs(P1-P8))
                    
                    if CN == 1.0: T[i, j] = 1
                    if CN == 3.0: B[i, j] = 1

        # Filtro de minucias
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                if T[i,j]: T[i,j] = not (T[i-1:i+2,j-1:j+2].sum() > 1)
                if B[i,j]: B[i,j] = not (B[i-1:i+2,j-1:j+2].sum() > 1)
        
        return T, B # Devuelve los mapas de terminaciones y bifurcaciones

if __name__ == "__main__":
    root = tk.Tk()
    app = FingerprintApp(root)
    root.mainloop()