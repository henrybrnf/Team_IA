import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2

class FingerprintApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Analizador de Huellas Dactilares - Proceso Detallado")
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
        self.lbl_binarized = self._create_image_panel("2. Imagen Binarizada", 0, 1)
        self.lbl_thinned = self._create_image_panel("3. Adelgazamiento (Esqueleto)", 1, 0)
        self.lbl_final = self._create_image_panel("4. Resultado (Minucias Marcadas en Esqueleto)", 1, 1)
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
            if image_data.dtype == bool:
                image_data = (image_data * 255).astype(np.uint8)
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
        
        img_gray = self.original_pil_image.convert("L")
        self.n, self.m = img_gray.size 
        img_np = np.array(img_gray)
        
        binarizado = self.binarizar(img_np, 132)
        self._update_image_label(self.lbl_binarized, binarizado)
        self.master.update_idletasks()
        
        adelgazado = self.adelgazar(binarizado.copy())
        adelgazado = self.adelgazar(adelgazado)
        self._update_image_label(self.lbl_thinned, adelgazado)
        self.master.update_idletasks()
        
        # --- LLAMADA: ---
        img_remarcada, terminaciones, bifurcaciones = self.analizar_y_remarcar(adelgazado)
        self._update_image_label(self.lbl_final, img_remarcada)
        
        self.lbl_results.config(text=f"ANÁLISIS COMPLETO  |  Terminaciones: {terminaciones}  |  Bifurcaciones: {bifurcaciones}", font=("Helvetica", 12, "bold"))

    def binarizar(self, Img, u):
        return Img >= u

    def adelgazar(self, Img):
        for i in range(1, self.m - 1):
            for j in range(1, self.n - 1):
                if Img[i, j]:
                    R = Img[i-1:i+2, j-1:j+2].astype(int)
                    if R[1,1] == 1:
                        should_be_removed = (self.thining1(R) or self.thining2(R) or
                                             self.pruning1(R) or self.pruning2(R) or
                                             self.caso1(R) or self.caso2(R) or
                                             self.caso3(R) or self.caso4(R))
                        Img[i, j] = int(not should_be_removed)
        return Img

    # --- FUNCIÓN ---
    def analizar_y_remarcar(self, Img):
        # La función ahora solo necesita la imagen adelgazada (Img)
        T = np.zeros((self.m, self.n))
        B = np.zeros((self.m, self.n))
        for i in range(1, self.m - 1):
            for j in range(1, self.n - 1):
                if Img[i, j]:
                    R = Img[i-1:i+2, j-1:j+2].astype(int)
                    P1,P2,P3 = int(R[0,0]),int(R[0,1]),int(R[0,2])
                    P8,P, P4 = int(R[1,0]),int(R[1,1]),int(R[1,2])
                    P7,P6,P5 = int(R[2,0]),int(R[2,1]),int(R[2,2])
                    CN = 0.5 * (abs(P2-P1) + abs(P3-P2) + abs(P4-P3) + abs(P5-P4) + abs(P6-P5) + abs(P7-P6) + abs(P8-P7) + abs(P1-P8))
                    if CN == 1.0: T[i, j] = 1
                    if CN == 3.0: B[i, j] = 1
        for i in range(1, self.m - 1):
            for j in range(1, self.n - 1):
                if T[i,j]: T[i,j] = not (T[i-1:i+2,j-1:j+2].sum() > 1)
                if B[i,j]: B[i,j] = not (B[i-1:i+2,j-1:j+2].sum() > 1)
        
        terminacion, bifurcacion = 0, 0
        
        # --- Crear el lienzo a partir del esqueleto ---
        esqueleto_uint8 = (Img * 255).astype(np.uint8)
        img_para_dibujar = cv2.cvtColor(esqueleto_uint8, cv2.COLOR_GRAY2RGB)
        
        for i in range(1, self.m - 1):
            for j in range(1, self.n - 1):
                if T[i, j]:
                    terminacion += 1
                    cv2.rectangle(img_para_dibujar, (j-3, i-3), (j+3, i+3), (255, 0, 0), 1) # Rojo
                if B[i, j]:
                    bifurcacion += 1
                    cv2.circle(img_para_dibujar, (j, i), 4, (0, 0, 255), 1) # Azul
        
        return img_para_dibujar, terminacion, bifurcacion
    
    def parathining1(self, R): return R[0,:].sum() == 0 and R[2,:].sum() == 3
    def thining1(self, R): return self.parathining1(R) or self.parathining1(np.rot90(R, k=1)) or self.parathining1(np.rot90(R, k=2)) or self.parathining1(np.rot90(R, k=3))
    def parathining2(self, R): return (R[1,0] + R[2,1]) == 2 and (R[0,1] + R[0,2] + R[1,2]) == 0
    def thining2(self, R): return self.parathining2(R) or self.parathining2(np.rot90(R, k=1)) or self.parathining2(np.rot90(R, k=2)) or self.parathining2(np.rot90(R, k=3))
    def parapruning1(self, R): return (R[1,0] + R[0,:].sum() + R[:,2].sum()) == 0
    def pruning1(self, R): return self.parapruning1(R) or self.parapruning1(np.rot90(R, k=1)) or self.parapruning1(np.rot90(R, k=2)) or self.parapruning1(np.rot90(R, k=3))
    def parapruning2(self, R): return (R[:,0].sum() + R[0,:].sum() + R[1,2]) == 0
    def pruning2(self, R): return self.parapruning2(R) or self.parapruning2(np.rot90(R, k=1)) or self.parapruning2(np.rot90(R, k=2)) or self.parapruning2(np.rot90(R, k=3))
    def caso1(self, R): return (R[1,0]+R[2,1])==2 and (R[0,1]+R[1,2])==0
    def caso2(self, R): return (R[1,0]+R[0,1])==2 and (R[2,1]+R[2,2]+R[1,2])==0
    def caso3(self, R): return (R[1,0]+R[2,1])==0 and (R[0,1]+R[1,2])==2
    def caso4(self, R): return (R[0,0]+R[0,1]+R[1,0])==0 and (R[2,1]+R[1,2])==2

if __name__ == "__main__":
    root = tk.Tk()
    app = FingerprintApp(root)
    root.mainloop()