### Ce programme teste le démélange selon les algorithmes
### classiques présentés dans [MBCGGPA14].
import numpy as np
import matplotlib.pyplot as plt
from algo import sc_n_findr, affcoord

# * Paramètres
M = 10000
N = 20
K = 3
A0 = np.random.uniform(-10, 10, (N, K))
S0 = np.random.dirichlet(np.ones(K), M).T
Y = A0 @ S0


# * Traitement
A = sc_n_findr(Y, K)                   # Récup. simplexe
S = affcoord(A, Y)
SA = affcoord(A, A)
  
print(f'rang de A0: {np.linalg.matrix_rank(A0)}')
print(f'rang de A: {np.linalg.matrix_rank(A)}')
print(f'verif coord aff: {np.allclose(SA, np.eye(K))}')


# * Affichage
plt.figure()
H = np.array([[1, .5], [0, np.sqrt(3)/2]])
for k in range(K//2):
  plt.subplot(1,K//2,1+k)
  R = H @ S[2*k:2*k+2,:]
  plt.scatter(R[0,:], R[1,:])
  RA = H @ SA[2*k:2*k+2,:]
  plt.scatter(RA[0,:], RA[1,:], c='r')

plt.show(block=False)
