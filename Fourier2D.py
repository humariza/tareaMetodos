import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq, ifft, fft2, ifft2, fftshift, ifftshift
from matplotlib.colors import LogNorm
#Importando imagen
porFiltrar= plt.imread('arbol(1).png')

x,y=porFiltrar.shape
fourier2d=fft2(porFiltrar)
centrada = abs(fftshift(fourier2d)) #absolute para ponerla en escala log

plt.figure()
plt.title('TF Arbol')
plt.imshow(np.log(centrada))
plt.colorbar()
plt.savefig('ArizaHumberto_FT2D.pdf')



maximo =np.max(centrada[::][::]) #Rango de frecuencias
# print(maximo)

centrada2 = fftshift(fourier2d)

def filtro2d(maximo,prueba,centrada2):
    for i in range(len(centrada2)):
        for j in range(len(centrada2)):
            if(centrada2[i][j]>prueba and centrada2[i][j]<maximo): 
                centrada2[i][j]=0
    
    return centrada2
#esto debería funcionar como filtro

filt=filtro2d(maximo,2100,centrada2) #parece que 2100 es una de los mejores resoluciones que queda

plt.figure()
plt.title("TF filtrada")
plt.imshow(np.log(abs(filt)),norm=LogNorm())
plt.colorbar()
plt.savefig("ArizaHumberto_FT2D_filtrada.pdf.")

#cmap gray para que quede igual al color del arbol
centrada2=ifftshift(filt)
invertida = ifft2(centrada2)
plt.figure()
plt.title("filtrada")
plt.imshow(abs(invertida), cmap='gray')
plt.savefig("ArizaHumberto_Imagen_filtrada.pdf.")

