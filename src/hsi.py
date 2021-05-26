### Librairie d'utilitaire pour manipuler les images
### hyper-spectrales.
import numpy as np
import spectral as S


def hsi2rgb(lambdas, cube, R=611.3, G=549.2, B=464.3):
  """Construit une image RGB à partir d'un cube hyper-spectral
  et des longueurs d'ondes. Les trois longueurs d'ondes
  sélectionnées sont les plus proches des longueurs R, G et
  B. Chaque canal est codé sur 256 niveaux avec une
  normalisation au préalable.
  """
  index_R = np.argmin(np.abs(lambdas - R))
  index_G = np.argmin(np.abs(lambdas - G))
  index_B = np.argmin(np.abs(lambdas - B))

  img = cube[..., (index_R, index_G, index_B)]
  img -= img.min()
  img /= img.max()
  return (255*img).astype(np.uint8)


def hsiread(filename):
  """Retourne les longueurs d'ondes et le cube de données
  hauteur x largeur x profondeur de l'image hyper-spectrale
  du fichier 'filename'.
  """
  img = S.open_image(filename)          # Lecture des données
  lambdas = np.array(img.bands.centers) # Centre des bandes [nm]
  cube = img.load(dtype=np.float32)     # En tableau np flottant
  return lambdas, np.array(cube)
