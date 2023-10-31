# ml_project

Machine learning and differentiable programming project

## TODO

- Install GPU => [Install CUDA WSL2](https://www.youtube.com/watch?v=R4m8YEixidI)

## Projet : Generer des descriptions d'images

À partir du dataset Flickr contenant 8000 images avec leurs descriptions en anglais, notre objectif est d'étudier les différents modèles possibles pour effectuer cette tâche comprenant LSTM, CNN, RNN ainsi que ConvNet et de les comparer en terme de temps de convergence, de précision et de complexité.

## Env Romain

### Conda

```bash
conda create --name venv38 python=3.8
conda activate venv38
conda install pandas numpy pytorch torchvision pillow tqdm matplotlib
# For using ipykernel in VSCode
conda install -n venv38 ipykernel --update-deps --force-reinstall
# For GPU and cuda 12.3 (romain)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

If you have an error with ipywidgets, try this:

```bash
# ImportError: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
conda install -c conda-forge ipywidgets
```

## Log book

### Removing 5 images not in the dataset

```python
# Read the file containing the captions
df = pd.read_csv("Flickr8k_text/Flickr8k.token.txt", sep='\t', header=None)
df.rename(columns={0:'image', 1:'caption'}, inplace=True)
display(df.head())

print("Length of df: ", len(df))

# Remove #digit from image name
df['image'] = df['image'].apply(lambda x : x.split('.jpg')[0] + '.jpg')
display(df.head())
print("Number of unique images: ", len(df['image'].unique()))

# Get all images' names in Flicker8k_Dataset
img_names = glob.glob("Flicker8k_Dataset/*.jpg")
print("Number of images in Flicker8k_Dataset: ", len(img_names))

img_names = [os.path.basename(img_name) for img_name in img_names]
for i in range(5):
    print(img_names[i])

# Remove lines in df that are not in Flicker8k_Dataset
print("Length of df: ", len(df))
df = df[df['image'].isin(img_names)]
print("Length of df: ", len(df))
```
