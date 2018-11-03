import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq, ifft, fft2, ifft2, fftshift, ifftshift


#Importando imagen
porFiltrar= plt.imread('arbol(1).png')

x,y=porFiltrar.shape
fourier2d=fft2(porFiltrar)
centrada = abs(fftshift(fourier2d))

plt.figure()
plt.title('TF Arbol')
plt.imshow(np.log(centrada))
#esto deberia tener una escala o alg asi respecto a los colores pero estoy sin tiempo
plt.savefig('ArizaHumberto_FT2D.pdf')