import numpy as np
import spectral as S
import matplotlib.pyplot as plt
from numba import njit



# * Utilitaires
def hsi2rgb(lambdas, cube):
  """Construit une image RGB à partir d'un cube hyperspectral
  et des longueurs d'ondes du cube.
  """
  lambda_R = 611.3                # Rouge en nm
  lambda_G = 549.2                # Vert en nm
  lambda_B = 464.3                # Bleu en nm

  index_R = np.argmin(np.abs(lambdas - lambda_R))
  index_G = np.argmin(np.abs(lambdas - lambda_G))
  index_B = np.argmin(np.abs(lambdas - lambda_B))

  img = cube[..., (index_R, index_G, index_B)]
  img -= img.min()
  img /= img.max()
  return (255*img).astype(np.uint8)


@njit
def hsi_dist(x, y):
  """Retourne la distance entre deux spectres x et y."""
  return np.linalg.norm(x - y, 1) / len(x)


@njit
def mean_shift_filter(cube, sr, lr, niter=100, tol=1e-5):
  height, width, nbands = cube.shape # Taille
  shifted = cube.copy()              # Résultat du filtrage

  for y in range(height):
    for x in range(width):
      xc, yc = x, y
      valc = cube[y, x, :]
    
      for _ in range(niter):
        n = 0
        newxc, newyc = 0, 0
        newvalc = np.zeros_like(valc)
        
        ymin = max(0, yc - sr)
        ymax = min(yc + sr, height)
        xmin = max(0, xc - sr)
        xmax = min(xc + sr, width)
        
        for y2 in range(ymin, ymax):
          for x2 in range(xmin, xmax):
            if hsi_dist(valc, cube[y2, x2, :]) < lr:
              n += 1
              newxc += x2
              newyc += y2
              newvalc += cube[y2, x2, :]

        newyc = int(newyc/n + 0.5)
        newxc = int(newxc/n + 0.5)
        newvalc /= n
        
        if newxc == xc and newyc == yc and hsi_dist(valc, newvalc) < tol:
          break
        xc, yc = newxc, newyc
        valc = newvalc
  
      shifted[y, x] = valc
  return shifted


# * Segmentation
def hsisegment(cube):
  height, width, nbands = cube.shape # Taille
  
  classes, segments = np.unique(np.around(np.vstack(cube), 5),
                                return_inverse=True,
                                axis=0)      
  return segments.reshape((height, width))



# * Paramètres
filename = '../data/Reflectance/image_bs_2_40.bip.hdr'
img = S.open_image(filename)          # Lecture des données
lambdas = np.array(img.bands.centers) # Centre des bandes [nm]
height, width, depth = img.shape      # Dimensions du cube
cube = img.load(dtype=np.float32)     # En tableau np flottant
cube = np.array(cube)
cube = cube[:100, :100]


# * Normalisation
mus = cube.mean(axis=(0,1))
sigmas = cube.std(axis=(0,1))
cube = (cube - mus)/sigmas


# * MeanShiftAlgorithm et segmentation
shifted = mean_shift_filter(cube, 10, .3)
segments = hsisegment(shifted)
print(f'Nombre de segments: {1+segments.max()}')



# * Affichage
plt.subplot(131)
plt.imshow(hsi2rgb(lambdas, mus + sigmas * cube))
plt.subplot(132)
plt.imshow(hsi2rgb(lambdas, mus + sigmas * shifted))
plt.subplot(133)
plt.imshow(segments, cmap='tab20c')
plt.show(block=False)
