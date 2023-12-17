# Rapport de mini-projet : Génération automatique de descriptions d'images à l'aide réseau ResNet et d'un Transformers

## Introduction

La génération automatique de descriptions d'images est une tâche complexe qui implique la compréhension de contenu visuel et la production de texte descriptif. Ce projet vise à explorer l'utilisation d'un réseau ResNet et d'un modèle Transformers pour accomplir cette tâche. Le réseau ResNet 18 a été choisi comme base pour extraire des caractéristiques visuelles en raison de son architecture profonde et de sa capacité à capturer des informations complexes.

Pour ce projet, nous nous sommes basés sur le concours kaggle suivant : [Kaggle Project](https://www.kaggle.com/datasets/adityajn105/flickr8k). En s’inspirant des différents modèles soumis pour ce concours, nous avons pour objectif d’étudier les impacts du dropout, du learning rate, du nombre d’epochs et d’autres paramètres, sur la qualité des descriptions en sortie du modèle. (+ accuracy + Scaling law ??)

## Méthode

### L'architecture du Resnet 18
![Resnet 18](https://penseeartificielle.fr/wp-content/uploads/2019/01/Proposed-Modified-ResNet-18-architecture-for-Bangla-HCR-In-the-diagram-conv-stands-for.jpg)

Le choix du modèle ResNet 18 s'explique par sa capacité à apprendre des représentations hiérarchiques profondes, ce qui est essentiel pour la tâche de génération de descriptions d'images. Cette architecture a démontré des performances solides sur divers corpus, ce qui en fait un choix approprié pour notre objectif.

### Fonction de coût

Dans un modèle de génération de séquences, comme celui utilisé pour générer des descriptions d'images, la Cross Entropy Loss est une fonction de coût couramment utilisée. Elle mesure la divergence entre la distribution de probabilité prédite par le modèle et la distribution de probabilité réelle des mots dans la séquence cible.

Cependant, dans le contexte des séquences de longueurs variables, une complication survient. Les séquences de mots ont différentes longueurs, et lors de l'entraînement d'un modèle, il est nécessaire d'aligner les prédictions du modèle avec les mots réels dans la séquence cible.

C'est là qu'intervient le masquage. Lorsque les séquences ont des longueurs variables, des tokens <pad> sont souvent ajoutés à la fin des séquences plus courtes pour les égaler en longueur. L'idée est de masquer ces tokens <pad> lors du calcul de la perte. En d'autres termes, on ne veut pas que le modèle soit pénalisé pour prédire correctement les tokens <pad>.

### Métrique d'évaluation

Lorsque vous évaluez un modèle de génération automatique de descriptions d'images, il est important de mesurer à quel point les descriptions générées sont similaires aux descriptions humaines de référence. C'est là que les métriques d'évaluation, telles que la BLEU score, METEOR, et ROUGE, entrent en jeu.

Notre métrique d'évaluation sera le BLEU Score (Bilingual Evaluation Understudy). Le BLEU score mesure la similarité entre la séquence de mots générée par le modèle et les références humaines. Elle évalue la précision du modèle en comparant les n-grammes (séquences de n mots consécutifs) générés avec ceux des références humaines. Une BLEU score plus élevée indique une meilleure concordance.

## Dataset

Pour ce projet, nous nous sommes basés sur le concours kaggle suivant : [Kaggle Project](https://www.kaggle.com/datasets/adityajn105/flickr8k). Il est composé de 8000 images diverses d'environ 500x300 et leurs descriptions associées, avec 5 descriptions différentes par image.

Le Dataset Flickr8k a été manuellement sélectionné pour que n’apparaissent pas de personnes ou de lieux connus. Il y a en tout 8096 images et 40460 descriptions en anglais, pour au total un vocabulaire de 8360 mots.

## Implémentation  et analyse des résultats

### Implémentation

La première étape du code consiste à réaliser un Data Cleaning permettant de rendre le texte plus facile à traiter par le modèle (retirer les chiffres, les mots à une seule lettre, ajouter 'start' et 'end' pour uniformiser le format des descriptions).
Le dataset est ensuite découpé en train et validation datasets et convertis en DataLoader. 

Le ResNet est utilisé, dans notre cas, pour récupérer les vecteurs de features, c'est-à-dire la représentation de l'image en un vecteur de longueur 512 la caractérisant. Cette méthode est très utilisée car elle permet d'empêcher le "vanishing gradient" (récurrent dans les modèles traitant des images à cause du grand nombre de couches) et qu'il s'agit d'un modèle pré-entraîné sur un grand nombre d'images.

Une fois les données mises au bon format pour le modèle Transformer, nous utilisons des fonctions sinusoidales et cosinusoidales pour coder les positions dans une séquence de caractères.

La partie permettant de traduire la sortie du ResNet en texte décrivant l'image est un décodeur à 4 couches auquel est ajouté une couche d'embedding et une couche linéaire permettant de transformer la sortie du décodeur en un vecteur de scores pour chaque mot de vocabulaire et de convertir les indices de mots en embedding.

Enfin, le modèle est entraîné sur une plage d'epochs variant entre 30 et 50 et évalué grâce à la métrique du score bleu.

### Analyse des résultats

Notre première analyse concerne la loss et ses variations selon le nombre d'epochs. Nous avons tout d'abord remarqué que la training loss stagnait à partir de 30 epochs et que la validation loss dépasse la training loss à partir de la 5e epoch. En souhaitant mieux comprendre cette évolution, nous avons réalisé un test avec 50 epochs et avons vu que la validation loss n'évoluait pas tandis que la train loss convergeait. Notre modèle n'overfit donc pas puisque la validation loss ne diverge pas.

Afin d'évaluer l'impact du nombre de paramètres sur l'overfitting, nous avons modifié le nombre de batch et avons observé que la convergence ne se faisait pas de la même façon : lorsque le nombre de batch est plus faible, la training loss converge vers une valeur plus basse tandis que la validation loss remonte après une vingtaine d'epochs. Cela prouve bien qu'augmenter le nombre de paramètres provoque le phénomène d'overfitting et on pourrait s'attendre à voir une double descente si on prolonge le nombre d'epochs.

Nous avons ensuite cherché à savoir si cette convergence de la loss était parallèle à celle du score bleu et donc de l'accuracy de notre modèle. En traçant la coube d'accuracy sur 30 epochs, nous avons obtenu ceci :

## Conclusion


