import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq, ifft
from scipy.interpolate import interp1d


#importando set de datos
signal=np.genfromtxt("signal.dat", delimiter=',')
incompletos=np.genfromtxt("incompletos.dat", delimiter=',')

incompletosx= incompletos[:,0]
incompletosy= incompletos[:,1]
senalx= signal[:,0]
senaly= signal[:,1]


#graficas
plt.figure()
plt.plot(senalx,senaly)
plt.title('Señal')
plt.savefig("ArizaHumberto_signal.pdf")

N=len(senalx)
#fourier
def fourier(senalx, senaly,N): 
    sumaTotal=np.zeros(N)
    for i in range(N):
        for j in range(N):
            sumaTotal[i]+=senaly[j]*np.exp(-1j*2*np.pi*j*(i/N))
    return sumaTotal/N

paso=senalx[1]-senaly[0]
#obteniendo la frecuencia de los datos
frecuencia=fftfreq(N,paso)
print('use fftfreq')

fouriersignal= fourier(senalx,senaly,N);
# print(fouriersignal)
plt.figure()
plt.plot(frecuencia,fouriersignal)
plt.title("Fourier Signal") 
plt.xlabel('Frecuencia')
plt.ylabel('Transformada de Fourier Discreta')
plt.show()
plt.savefig("ArizaHumberto_TF.pdf")



#sacando filtro de S7C2
def filtro(frecuencia,coeficientes):
	
	dataFinal=[]
	for i in range(len(frecuencia)):
		if(coeficientes[i]>(-0.025) and coeficientes[i]<0.025):
			dataFinal.append(frecuencia[i])
		
	return dataFinal

frecuenciasPrincipales = filtro(frecuencia,fouriersignal)
print('las frecuencias principales son: ########',frecuenciasPrincipales)

def filtro2(frecuencia,coeficientes):
	
	dataFinal=[]
	for i in range(len(frecuencia)):
		if(frecuencia[i]>1000 and frecuencia[i]<1000 ):
			dataFinal.append(0)
		else:
			dataFinal.append(coeficientes[i])
	return dataFinal
filtro= ifft(filtro2(frecuencia,fouriersignal))
plt.title('filtro')
plt.scatter(frecuencia,filtro,label='Filtro')
plt.ylabel("Filtro")
plt.xlabel("frecuencia")
plt.show()