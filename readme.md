# PSENet: Shape Robust Text Detection with Progressive Scale Expansion Network

Implémentation de "PSENet: Shape Robust Text Detection with Progressive Scale Expansion Network" (https://arxiv.org/abs/1806.02559), en Tensorflow 2.0.


Permet de générer un ensemble de données sous forme de tfRecord, puis d'entraîner un réseau de neurones destiné à prédire les masques de prédiction des zones de textes ( à plusieurs échelles ). 

Lors de la prédiction, l'algorithme pse est utilisé pour reconstruire les ground truth à partir des masques de prédiction à la plus petite échelle, ce qui permet de bien différencier les lignes de texte pour lesquelles les bounding box se touchent.

Exemple de résultat sur des documents textuels et un permis de conduire, faisant partie du jeu de données de test : 
![alt text](https://github.com/jlhervy/textBoundingBoxDetection/blob/master/img/exemples.png)
![alt text](https://github.com/jlhervy/textBoundingBoxDetection/blob/master/img/permis.png)

Ce repository est une des composantes de la solution que j'ai développée pour effectuer de la RAD-LAD, les composantes etant : classification selon des critères graphiques, détection des bounding box, lecture des zones extraites). Un script appelant permet de faire une prédiction "End to end", c'est à dire qu'il prend des images en entrée, et output le résultat de la classification, de la détection des bounding boxs, et de la lecture. Pour entraîner les réseaux de neurones, il y a un travail d'annotation et d'augmentation de données à effectuer. (notamment, grâce à la librairie ImgAug et l'outil OpenCV CVAT).
