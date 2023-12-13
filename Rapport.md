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

## Implémentation Analyses des résultats

### Implémentation

### Analyse des résultats

### Visualisation des résultats

## Conclusion

## Bibliographie
