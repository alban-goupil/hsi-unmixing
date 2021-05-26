# hsi-unmixing

## Introduction

Ce projet implémente en Python des algorithmes de dé-mélange
(unmixing) pour les images hyperspectrales. Le but étant de
trouver de manière non supervisée les spectres des
différents éléments de l'image, dans notre cas des résidus
de culture et du sol, pour faire ensuite une classification
des pixels.

## Fichiers

Les fichiers dans le répertoire [data](./data) sont les
entêtes de quelques images hyperspectrales qui servent aux
tests. Les données en elles-même ne sont pas dans ce dépôt.

Les fichiers du répertoire [src](./src) sont
+ [unmix-spa-sc_n_findr.py](./src/unmix-spa-sc_n_findr.py)
  compare les deux méthodes SPA et SC N FINDR sur des images
  hyperspectrales de test;
+ [hsi.py](./src/hsi.py) fournit des fonctions pour lire et
  afficher des images hyperspectrales;
+ [algo.py](./src/algo.py) implémente les algorithmes de
  dé-mélange et des fonctions d'aide pour les coordonnées
  barycentriques;
+ [meanshiftfilter.py](./src/meanshiftfilter.py) fait une
  segmentation en utilisant le filtrage "Mean Shift
  Algorithm";
+ [watershed.py](./src/watershed.py) et
  [contour.py](./src/contour.py) viennent de ce
  [site](https://www.pyimagesearch.com/2015/11/02/watershed-opencv/)
  pour faire de la segmentation par watershed;
  
Le répertoire [test](./test/) contient des scripts de test
des différents algorithmes.
