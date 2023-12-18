# ml_project

Machine learning and differentiable programming project

## Project: Image Captioning

La génération automatique de descriptions d'images est une tâche complexe qui implique la compréhension de contenu visuel et la production de texte descriptif. Ce projet vise à explorer l'utilisation d'un réseau ResNet (encodeur) et d'un modèle Transformers (décodeur) pour accomplir cette tâche. 

Pour ce projet, nous nous sommes basés sur le concours kaggle suivant : [Kaggle Project](https://www.kaggle.com/datasets/adityajn105/flickr8k). En s’inspirant des différents modèles soumis pour ce concours, nous avons pour premier objectif d’étudier l'impact des hyperparamètres sur les performances du modèle ainsi que de voir si les scaling laws s'appliquent bien à notre modèle.

**Voir `Rapport.md` pour plus d'explications.**

## Env

- Install GPU => [Install CUDA WSL2](https://www.youtube.com/watch?v=R4m8YEixidI)

### Conda

```bash
conda create --name venv38 python=3.8
conda activate venv38
conda install pandas numpy pytorch torchvision pillow tqdm matplotlib
# For using ipykernel in VSCode
conda install -n venv38 ipykernel --update-deps --force-reinstall
# For GPU and cuda 12.3 (me)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

If you have an error with ipywidgets, try this:

```bash
# ImportError: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
conda install -c conda-forge ipywidgets
```
