import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Moddèle 
E=168e9
L=18.3
b=3
h=0.06
Sec=b*h
Iz=b*h**3/12

rho=2330
nu=0.28

Force_ext=25

# finit element
nelmt=1


L_elmt=L/nelmt


# Pour 1 element 
Valp=np.sqrt((12*E*Iz/(L_elmt**3))/((rho*Sec*L_elmt/420)*156))/(2*np.pi)

# Reponse en frequence

w=2*np.pi*np.linspace(0.05, 0.5,7000)

ceq=0.000317*(12*E*Iz/(L_elmt**3))

zw=(12*E*Iz/(L_elmt**3))+ceq*1j*w-((rho*Sec*L_elmt/420)*156)*w**2

X1=Force_ext/zw

##############################
##############################
##############################
# Pour 10 elements
#w=2*np.pi*np.linspace(1, 5000,7000)

nelmt=10
nddl=2
noeud=2

Nr=nelmt*noeud*(nddl-1)+2

L_elmt=L/nelmt

k_elmt=E*Iz/(L_elmt**3)*np.array([[12,6*L_elmt,-12,6*L_elmt],[6*L_elmt,4*L_elmt**2,-6*L_elmt,2*L_elmt**2],[-12,-6*L_elmt,12,-6*L_elmt],[6*L_elmt,2*L_elmt**2,-6*L_elmt,4*L_elmt**2]])
m_elmt=(rho*Sec*L_elmt/420)*np.array([[156,22*L_elmt,54,-13*L_elmt],[22*L_elmt,4*L_elmt**2,13*L_elmt,-3*L_elmt**2],[54,13*L_elmt,156,-22*L_elmt],[-13*L_elmt,-3*L_elmt**2,-22*L_elmt,4*L_elmt**2]])


#Assemblage
Ktot=np.zeros([Nr,Nr])
Mtot=np.zeros([Nr,Nr])
for i in range(0,Nr-2,2):
    print(i)
    Ktot[i:i+4,i:i+4]=Ktot[i:i+4,i:i+4]+k_elmt
    Mtot[i:i+4,i:i+4]=Mtot[i:i+4,i:i+4]+m_elmt


# Conditions limites 
Mfin=Mtot[2:Nr,2:Nr]
Kfin=Ktot[2:Nr,2:Nr]

# Valeur propre

eigenvalues, PP=linalg.eig(Kfin,Mfin)

Valp_10=np.sqrt(eigenvalues)/(2*np.pi)


# Vecteur effort 
Ffin=np.zeros([Nr-2,1])
Ffin[-2,0]=Force_ext

# Amortissement 
Cfin=0.0317*Kfin

Xsol=np.zeros([len(w),1],dtype=complex)
# Reponse en frequence arround first mode
for ii in range(len(w)):
    Zww=-Mfin*w[ii]**2+1j*w[ii]*Cfin+Kfin
    X2=np.dot(linalg.inv(Zww),Ffin)
    Xsol[ii]=X2[-2,0]
    phase=-np.arctan2( np.imag(Xsol[ii]),np.real(Xsol[ii]))



## Diagram de Nyquist 
fig = plt.figure(1)
#plt.plot(w, np.imag(Xsol))
#ax = fig.add_subplot(111, projection='3d')
ax = plt.axes(projection ='3d')
ax.plot(w, np.real(Xsol[:,0]),np.imag(Xsol[:,0]),label='Nyquist 3D')

ax.plot(w, np.imag(Xsol[:,0]),zs=-0.12,zdir='y', label='Nyquist 3D')
ax.plot(w, np.real(Xsol[:,0]),zs=-0.12,zdir='z', label='Nyquist 3D')
ax.plot(np.real(Xsol[:,0]), np.imag(Xsol[:,0]),zs=0.1,zdir='x', label='Nyquist 3D')
plt.grid()
ax.set_xlabel('Freq')
ax.set_zlabel('Im')
ax.set_ylabel('Re')
ax.set_title('Diagramme de Nyquist en 3D')
plt.show()




