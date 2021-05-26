### Librairie d'algorithmes de base pour l'analyse d'images
### hyper-spectrales.
import numpy as np
from numpy.linalg import norm, matrix_rank
from scipy.linalg import orth, lstsq, null_space


def asf(Y, N):
  """Réduit la dimension de Y par un 'Affine Set Fitting'
  (AFS) qui est équivalent à une PCA. Y de taille
  n x m décrit m points de R^n. asf retourne X, C et d tels
  que Y = C @ X + d et minimisant |Y - (C @ X + d)|. C est
  orthogonale, ainsi X = C.T @ (Y - d).

  """
  phi = np.cov(Y, bias=True)
  w, v = np.linalg.eigh(phi)
  C = v[:,-N:]
  d = np.mean(Y, axis=1)
  X = C.T @ (Y - d[:, None])

  return X, C, d



def spa(Y, K):
  """Récupère K sommets du simplexe englobant les données
  Y grâce à la méthode du 'Successive Projections Algorithm'
  (SPA). Y de taille n x m décrit m points de R^n.
  """
  N, M = Y.shape
  Y = np.vstack((np.ones((1, M)), Y))
  
  P = np.eye(N+1)   # Projecteur sur l'espace orthogonal aux A
  A = np.zeros((N+1, K))          # Sommets
  for k in range(K):
    l = np.argmax(norm(P @ Y, axis=0))
    A[:, k] = Y[:, l]
    Pa = P @ A[:, k]
    P = (np.eye(N+1) - np.outer(Pa, Pa)/np.dot(Pa,Pa)).dot(P)

  return A[1:,:]



def sc_n_findr(Y, K, niter=100):
  """Récupère K sommets du simplexe de volume maximal
  englobant les données Y grâce à la méthode du 'SC
  N FINDR'. Les sommets appartiennent aux points décrits par
  Y qui est de taille n x m décrit m points de R^n.
  """
  N, M = Y.shape
  X = np.vstack((np.ones((1, M)), Y))
  B = np.zeros((1+N, K))

  # Recherche d'une base de rang plein
  assert matrix_rank(X) >= K
  m = 0
  for k in range(K):
    while matrix_rank(B) <= k:
      B[:,k] = X[:,m]
      m += 1

  # Recherche des sommets de volume maximal
  for _ in range(niter):
    for k in range(K):
      F = B.copy()
      F[:, k] = 0
      P = null_space(F.T).T
      l = np.argmax(norm(P @ X, axis=0))
      B[:,k] = X[:,l]
    
  return B[1:,:]




def barycentric(A, Y):
  """Retourne les coordonnées barycentriques des colonnes de
  Y selon les points décrits par les colonnes de A. On
  suppose que les colonnes de A sont affinement
  indépendantes. Y est d'abord orthogonalement projeté sur
  ce plan affine avant d'en calculer les coordonnées.
  """

  # mu est le barycentres des points A qui est alors
  # translaté vers l'origine pour avec l'espace euclidien E
  # sous-jacent à l'espace affine. 
  mu = np.mean(A, axis=1)       # Barycentre de A
  E = orth(A - mu[:, None])     # Base orthogonale de E

  # On projette Y sur l'espace affine en le projetant le
  # translaté par -mu d'abord sur E puis en ajoutant mu.
  X = E @ E.T @ (Y - mu[:, None]) # Proj. de Y translaté sur E

  # Les coordonnées barycentriques sont alors obtenues par
  # inversion de A.
  S, _, _, _ = lstsq(A, mu[:, None] + X)

  return S
