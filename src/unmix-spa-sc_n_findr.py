### Ce programme teste le démélange selon les algorithmes
### classiques présentés dans [MBCGGPA14].
import numpy as np
import matplotlib.pyplot as plt
import hsi
from algo import asf, spa, sc_n_findr, barycentric


# * Paramètres
filename = '../data/Reflectance/image_bs_2_2m_40.bip.hdr'
N = 20                          # Réduction de dimension
K = 3                           # Taille du simplexe


# * Récupération du cube de données
lambdas, cube = hsi.hsiread(filename)

I = np.logical_and(lambdas > 400, lambdas < 1015)
lambdas = lambdas[I]
cube = cube[:, :, I]
height, width, depth = cube.shape

Y = cube.copy().reshape(-1, depth).T # Dataframe des spectres


# * Prétraitement
L0 = np.max(Y, axis=1)          # Le flux incident estimé
Y /= L0[:,None]                 # Pour ne gardant que la transmittance
Y /= Y.sum(0)                   # Pour supprimer le cos(theta)


# * Traitement SPA et SC N FINDR
X, C, d = asf(Y, N)             # Réduction de dimension

# Par SPA
Aspa = spa(X, K)            # Récup. simplexe
Sspa = barycentric(Aspa, X) # Et des coordonnées barycentriques
SAspa = barycentric(Aspa, Aspa)

# Par SC N FINDR
Asc = sc_n_findr(X, K)    # Récup. simplexe
Ssc = barycentric(Asc, X) # Et des coordonnées barycentriques
SAsc = barycentric(Asc, Asc)
  
  
# * Affichage
plt.figure()
plt.imshow(hsi.hsi2rgb(lambdas, cube))

plt.figure()
plt.subplot(1,2,1)
plt.plot(lambdas, C @ Asc + d[:,None])
plt.title('Pure pixel spectra (SC N FINDR)')
plt.subplot(1,2,2)
plt.plot(lambdas, C @ Aspa + d[:,None])
plt.title('Pure pixel spectra (SPA)')


plt.figure()
plt.title('Barycentric coordinates (SPA)')
for k in range(K//2):
  plt.subplot(1,K//2,1+k)
  plt.scatter(Sspa[2*k,:], Sspa[2*k+1,:])
  plt.scatter(SAspa[2*k,:], SAspa[2*k+1,:], c='r')

plt.figure()
plt.title('Barycentric coordinates (SC N FINDR)')
for k in range(K//2):
  plt.subplot(1,K//2,1+k)
  plt.scatter(Ssc[2*k,:], Ssc[2*k+1,:])
  plt.scatter(SAsc[2*k,:], SAsc[2*k+1,:], c='r')

  
plt.figure()
plt.title('Barycentric coordinates (SPA)')
for k in range(K):
  plt.subplot(2,2,1+k)
  plt.imshow(Sspa[k,:].reshape(height,width), cmap='seismic')
  plt.colorbar()
  
plt.figure()
plt.title('Barycentric coordinates (SC N FINDR)')
for k in range(K):
  plt.subplot(2,2,1+k)
  plt.imshow(Ssc[k,:].reshape(height,width), cmap='seismic')
  plt.colorbar()

plt.show(block=False)
