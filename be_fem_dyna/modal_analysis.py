import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt


# Moddèle 
E=168e9
L=10
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

w=2*np.pi*np.linspace(0, 50,700)

ceq=0.000317*(12*E*Iz/(L_elmt**3))

zw=(12*E*Iz/(L_elmt**3))+ceq*1j*w-((rho*Sec*L_elmt/420)*156)*w**2

X1=Force_ext/zw

##############################
##############################
##############################
# Pour 10 elements
#w=2*np.pi*np.linspace(1, 5000,7000)

nelmt=50
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
eigenvaluesH, PPH=linalg.eigh(Kfin,Mfin)


Valp_10=np.sqrt(eigenvalues)/(2*np.pi)


# Vecteur effort 
Ffin=np.zeros([Nr-2,1])
Ffin[-2,0]=Force_ext

# Amortissement 
Cfin=0.000317*Kfin

Xsol=np.zeros([len(w),1])
# Reponse en frequence
for ii in range(len(w)):
    Zww=-Mfin*w[ii]**2+1j*w[ii]*Cfin+Kfin
    X2=np.dot(linalg.inv(Zww),Ffin)
    Xsol[ii]=20*np.log10(abs(X2[-2,0]))


##############################
##############################
##############################
# Projection 

N_mode=1
PP_tronc=PP[:,-N_mode:]
K_mod=np.dot(np.dot(np.transpose(PP_tronc),Kfin),PP_tronc)
M_mod=np.dot(np.dot(np.transpose(PP_tronc),Mfin),PP_tronc)
F_mod=np.transpose(PP_tronc)@Ffin
C_mod=np.dot(np.dot(np.transpose(PP_tronc),Cfin),PP_tronc);

Xproj=np.zeros([len(w),1])
# Reponse en frequence
for ii in range(len(w)):
    Zwwproj=-M_mod*w[ii]**2+1j*w[ii]*C_mod+K_mod
    print(Zwwproj)
    X2_mod=linalg.inv(Zwwproj)@F_mod
    Uchamp=np.dot(PP_tronc, X2_mod)
    print(Uchamp[-2,0])
    Xproj[ii]=20*np.log10(abs(Uchamp[-2,0]))
    

    

##############################
plt.plot(w/2/np.pi,20*np.log10(abs(X1)))
plt.plot(w/2/np.pi,Xsol)
plt.plot(w/2/np.pi,Xproj)
plt.show()

