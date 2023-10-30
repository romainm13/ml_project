# ml_project

Machine learning and differentiable programming project

## Projet : Generer des descriptions d'images

À partir du dataset Flickr contenant 8000 images avec leurs descriptions en anglais, notre objectif est d'étudier les différents modèles possibles pour effectuer cette tâche comprenant LSTM, CNN, RNN ainsi que ConvNet et de les comparer en terme de temps de convergence, de précision et de complexité.

## Env Romain

- Python 3.10.12 with virtualenv
- `pip install -r requirements.txt`
- `pip install typing_extensions==4.7.1` (retrograde compatibility)

### Conda

```bash
conda create --name ml_project # create env
conda activate ml_project
conda install pandas numpy pytorch torchvision pillow tqdm matplotlib
```

```bash
conda create --name venv38 python=3.8
conda activate venv38
conda install pandas numpy pytorch torchvision pillow tqdm matplotlib
conda install -n venv38 ipykernel --update-deps --force-reinstall
```
