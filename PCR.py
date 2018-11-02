import numpy as np
import matplotlib.pylab as plt

datos= np.genfromtxt('WDBC.dat',delimiter=',',usecols=(3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31))

datosByM=np.genfromtxt('WDBC.dat',delimiter=",",usecols=2)

#MATRIZ DE COVARIANZA HECHA EN S6C2
def matrizCovarianza(datos):
	dim = np.shape(datos)[1]
	cov = np.zeros([dim, dim])
	for i in range(dim):
		for j in range(dim):
			promedio_i = np.mean(datos[:,i])
			promedio_j = np.mean(datos[:,j])
			cov[i,j] = np.sum((datos[:,i]-promedio_i) * (datos[:,j]-promedio_j))
	return cov


# print(prueba)
# print("=======================================================")
# print(pruebaMetodo)
eigenvalues,eigenvectores = np.linalg.eig(matrizCovarianza(datos))

#print(eigenvalues)
#print(eigenvectores)