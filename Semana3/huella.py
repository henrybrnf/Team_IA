import numpy as np
from PIL import Image
import cv2
n,m = 0,0
def binarizar(Img,u):
    return Img>=u
def parathining1(R):
    return R[0,:].sum()==0 and R[2,:].sum()==3
def thining1(R):
    return parathining1(R) or parathining1(np.rot90(R,k=1)) or parathining1(np.rot90(R,k=2)) or parathining1(np.rot90(R,k=3))
def parathining2(R):
    return (R[1,0]+R[2,1])==2 and (R[0,1]+R[0,2]+R[1,2])==0
def thining2(R):
    return parathining2(R) or parathining2(np.rot90(R,k=1)) or parathining2(np.rot90(R,k=2)) or parathining2(np.rot90(R,k=3))
def parapruning1(R):
    return (R[1,0]+R[0,:].sum()+R[:,2].sum())==0
def pruning1(R):
    return parapruning1(R) or parapruning1(np.rot90(R,k=1)) or parapruning1(np.rot90(R,k=2)) or parapruning1(np.rot90(R,k=3))
def parapruning2(R):
    return (R[:,0].sum()+R[0,:].sum()+R[1,2])==0
def pruning2(R):
    return parapruning2(R) or parapruning2(np.rot90(R,k=1)) or parapruning2(np.rot90(R,k=2)) or parapruning2(np.rot90(R,k=3))
def caso1(R):
    return (R[1,0]+R[2,1])==2 and (R[0,1]+R[1,2])==0
def caso2(R):
    return (R[1,0]+R[0,1])==2 and (R[2,1]+R[2,2]+R[1,2])==0
def caso3(R):
    return (R[1,0]+R[2,1])==0 and (R[0,1]+R[1,2])==2
def caso4(R):
    return (R[0,0]+R[0,1]+R[1,0])==0 and (R[2,1]+R[1,2])==2

def adelgazar(Img):
    for i in range(1,m-1):
        for j in range(1,n-1):
            if Img[i,j]:
                R = Img[i-1:i+2,j-1:j+2].astype(int)# obtengo un retazo de 3x3
                if R[1,1] == 1:
                    Img[i,j] = int(not (thining1(R) or thining2(R) or pruning1(R) or pruning2(R) or caso1(R) or caso2(R) or caso3(R) or caso4(R)))
    return Img            
def validar(Img):
    T = np.zeros((m,n))
    B = np.zeros((m,n))
    for i in range(1,m-1):
        for j in range(1,n-1):
            if Img[i,j]:
                R = Img[i-1:i+2,j-1:j+2].astype(int)
                P1,P2,P3 = int(R[0,0]),int(R[0,1]),int(R[0,2])
                P8,P, P4 = int(R[1,0]),int(R[1,1]),int(R[1,2])
                P7,P6,P5 = int(R[2,0]),int(R[2,1]),int(R[2,2])
                CN = 0.5*(abs(P2-P1)+abs(P3-P2)+abs(P4-P3)+abs(P5-P4)+abs(P6-P5)+abs(P7-P6)+abs(P8-P7)+abs(P1-P8))
                if CN==1.0:
                    T[i,j]=1
                if CN==3.0:
                    B[i,j]=1
    for i in range(1,m-1):
        for j in range(1,n-1):
            if T[i,j]:
                T[i,j] = not (T[i-1:i+2,j-1:j+2].astype(int).sum()>1)
            if B[i,j]:
                B[i,j] = not (B[i-1:i+2,j-1:j+2].astype(int).sum()>1)   
    return T,B  
def remarcar(Img):
    terminacion = 0
    bifurcacion = 0
    imgRGB = cv2.imread("test.jpg")# agregarle las minucias a la imagen
    T,B  = validar(Img)
    for i in range(1,m-1):
        for j in range(1,n-1):
            if T[i,j]:
                terminacion = terminacion + 1
                cv2.rectangle(imgRGB,(j,i),(j+2,i+2),(255,0,0),1)
            if B[i,j]:
                bifurcacion = bifurcacion + 1
                cv2.circle(imgRGB,(j,i),4,(0,0,255),1)
    print("Terminaciones: "+str(terminacion))
    print("Bifurcaciones : " + str(bifurcacion))
    imgRGB = cv2.resize(imgRGB, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_AREA)
    cv2.imshow('remarcado', imgRGB)
    cv2.imwrite("remarcado.jpg", imgRGB)  # guarda la imagen con minucias
    cv2.waitKey(0)   # espera a que presiones una tecla
    cv2.destroyAllWindows()


imgGray = Image.open("huella.tif").convert("L")
imgGray.show()
n,m = imgGray.size
imgNP = np.array(imgGray)
binarizado = binarizar(imgNP,132)
adelgazado = adelgazar(adelgazar(binarizado))
im = Image.fromarray(adelgazado)
im.save("adelgazado.tif")
im.convert("RGB").save("test.jpg")
remarcar(adelgazado)
im.show()