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

paso=senalx[1]-senalx[0]
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
# #plt.show()
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
		if(frecuencia[i]>1000 and frecuencia[i]<(-1000) ):
			dataFinal.append(0)
		else:
			dataFinal.append(coeficientes[i])
	return dataFinal
filtro= ifft(filtro2(frecuencia,fouriersignal))
plt.title('Filtro')
plt.plot(frecuencia,filtro,label='Filtro')
plt.ylabel("Filtro")
plt.xlabel("Frecuencia")
plt.legend()
plt.savefig("ArizaHumberto_filtrada.pdf")
# #plt.show()

#punto 3.7
print('Punto 3.7')
print( 'Los datos incompletos no presentan el mismo espacio entre datos, y debido a esto, no es posible calcular correctamete las frecuencias afectando este metodo ')
#punto 3.8
interpolacionCubica =interp1d(incompletosx, incompletosy, kind='cubic') 
interpolacionCuadratica = interp1d(incompletosx, incompletosy, kind='quadratic') 

minimx = min(incompletosx)
maximx = max(incompletosx)

#los 512 puntos de la interpolacion
puntos =np.linspace(minimx, maximx , 512)

cuadratica = interpolacionCuadratica(puntos) 
cubica = interpolacionCubica(puntos)


frecuenciaCubica = fftfreq(len(senalx),paso)
frecuenciaCuadratica=fftfreq(len(senalx),paso)

transformadaCubica = fourier(senalx,cubica,N)
transformadaCuadratica = fourier(senalx,cuadratica,N)

plt.figure()
plt.subplot(3,1,1)
plt.title('Interpolados(cuadratica,cubica)y Original')
plt.plot(frecuenciaCuadratica,transformadaCuadratica,label="cuadratica")
plt.xlabel('Frecuencia')
plt.ylabel('Transformada')
plt.legend()
plt.subplot(3,1,2)
plt.plot(frecuenciaCubica,transformadaCubica,label="cubica",color='red')
plt.xlabel('Frecuencia')
plt.ylabel('Transformada')
plt.legend()
plt.subplot(3,1,3)
plt.plot(frecuencia,fouriersignal,label="Original",color='green')
plt.xlabel('Frecuencia')
plt.ylabel('Transformada')
plt.legend()
#plt.show()
plt.savefig("ArizaHumberto_TF_interpola.pdf")



print("Las diferencias principales pueden verse en los valores de los segundos picos mas altos de la trnsofrmada de fourier, siendo mas grande los segundos picos mas altos de la original ,posteriormente la cuadratica y por ultimo la cubica. Adicionalmente existe mas 'ruido' en las interpolaciones extendiendose casi 1000 mas en las frecuencias")
def filtro500(frecuencia,coeficientes):
	
	dataFinal=[]
	for i in range(len(frecuencia)):
		if(frecuencia[i]>500 and frecuencia[i]<(-500) ):
			dataFinal.append(0)
		else:
			dataFinal.append(coeficientes[i])
	return dataFinal


cubico500 = filtro500(frecuenciaCubica,transformadaCubica)
cubico1000 = filtro2(frecuenciaCubica,transformadaCubica)
cuadratica500 = filtro500(frecuenciaCuadratica,transformadaCuadratica)
cuadratica1000=filtro2(frecuenciaCuadratica,transformadaCuadratica)
originales500 =filtro500(frecuencia,fouriersignal)
originales1000=filtro2(frecuencia,fouriersignal)
plt.figure()
plt.subplot(2,1,1)
plt.plot(frecuenciaCubica,cubico500,label="Filtro 500 cubico")
plt.plot(frecuenciaCuadratica,cuadratica500,label="Filtro 500 cuadratico")
plt.plot(frecuencia,originales500,label="Filtro 500 originales")
plt.legend()
plt.title("Filtros Originales e Interpoladas")
plt.subplot(2,1,2)
plt.plot(frecuenciaCubica,cubico1000,label="Filtro 1000 cubico")
plt.plot(frecuenciaCuadratica,cuadratica1000,label="Filtro 1000 cuadratico")
plt.plot(frecuencia,originales1000,label="Filtro 1000 originales")
plt.legend()
plt.savefig("ArizaHumberto_2Filtros.pdf")
###plt.show()
#NOTA : POR ALGUNA EXTRANA RAZON LOS FILTROS NO PARECEN APLICARSE PERO SI NOS VAMOS A LA DEFINICION PARECEN ESTAR BIEN