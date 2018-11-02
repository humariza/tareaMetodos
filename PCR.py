import numpy as np
import matplotlib.pylab as plt

datos= np.genfromtxt('WDBC.dat',delimiter=',',usecols=(3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31))

datosByM=np.genfromtxt('WDBC.dat',delimiter=",",usecols=1, dtype='|U16')

#MATRIZ DE COVARIANZA HECHA EN S6C2
def matrizCovarianza(datos):
	dim = np.shape(datos)[1]
	cov = np.zeros([dim, dim])
	for i in range(dim):
		for j in range(dim):
			promedio_i = np.mean(datos[:,i])
			promedio_j = np.mean(datos[:,j])
			cov[i,j] = np.sum((datos[:,i]-promedio_i) * (datos[:,j]-promedio_j))/(dim-1)
	return cov


# print(prueba)
# print("=======================================================")
# print(pruebaMetodo)
eigenvalues,eigenvectores = np.linalg.eig(matrizCovarianza(datos))

for i in range(len(eigenvalues)):
    print ('=============Autovalor=============', eigenvalues[i])
    print ('=============Autovector============', eigenvectores[i])
#print(eigenvalues)
#print(eigenvectores)
#PUNTO 1 D
print ('Los dos valores mas importantes corresponden a los autovalores mas grandes que son los que contienen mas correlacion que son:')
print (eigenvalues[0],'\n y',eigenvalues[1])

#punto 2D 
#se buscan los indices malos y buenos
buenos=[]
malos=[]
#Recorre todos los datos de tipo y busca B o M que son los unicos dos valores existentes y guarda la posicion en el tipo de dato para poder hacer la separacion
for indice in range(len(datosByM)):
   
    if (datosByM[indice]=="B"):
        buenos.append(indice)
    else:
        malos.append(indice)

#print(buenos)

#ahora con los indices anteriores separamos este set de datos en datos malignos y begninos
datosMalos=[]
datosBuenos=[]
for i in range(len(buenos)):
    datosBuenos.append(datos[buenos[i]])
for i in range(len(malos)):
    datosMalos.append(datos[malos[i]])
print(len(datosMalos))
print(len(datosBuenos))


eje1=np.matmul(datosMalos,eigenvectores)
eje2=np.matmul(datosBuenos,eigenvectores)
print(len(eje1))
print(len(eje2))
plt.figure()
plt.title("Datos Buenos y Malos")
plt.scatter(eje1[:,0],eje1[:,1], label="M", color="Red")
plt.scatter(eje2[:,0],eje2[:,1], label="B", color="Green")
plt.legend()
plt.show()