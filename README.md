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
# For GPU and cuda 12.3 (me)
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

### Error training at the end

```python
EPOCH = 30

ictModel = ImageCaptionModel(16, 4, vocab_size, 512).to(device)
optimizer = torch.optim.Adam(ictModel.parameters(), lr = 0.00001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.8, patience=2, verbose = True)
criterion = torch.nn.CrossEntropyLoss(reduction='none')
min_val_loss = np.float('Inf')

for epoch in tqdm(range(EPOCH)):
    total_epoch_train_loss = 0
    total_epoch_valid_loss = 0
    total_train_words = 0
    total_valid_words = 0
    ictModel.train()

    ### Train Loop
    for caption_seq, target_seq, image_embed in train_dataloader_resnet:

        optimizer.zero_grad()

        image_embed = image_embed.squeeze(1).to(device)
        caption_seq = caption_seq.to(device)
        target_seq = target_seq.to(device)

        output, padding_mask = ictModel.forward(image_embed, caption_seq)
        output = output.permute(1, 2, 0)

        loss = criterion(output,target_seq)

        loss_masked = torch.mul(loss, padding_mask)

        final_batch_loss = torch.sum(loss_masked)/torch.sum(padding_mask)

        final_batch_loss.backward()
        optimizer.step()
        total_epoch_train_loss += torch.sum(loss_masked).detach().item()
        total_train_words += torch.sum(padding_mask)

 
    total_epoch_train_loss = total_epoch_train_loss/total_train_words
  

    ### Eval Loop
    ictModel.eval()
    with torch.no_grad():
        for caption_seq, target_seq, image_embed in valid_dataloader_resnet:

            image_embed = image_embed.squeeze(1).to(device)
            caption_seq = caption_seq.to(device)
            target_seq = target_seq.to(device)

            output, padding_mask = ictModel.forward(image_embed, caption_seq)
            output = output.permute(1, 2, 0)

            loss = criterion(output,target_seq)

            loss_masked = torch.mul(loss, padding_mask)

            total_epoch_valid_loss += torch.sum(loss_masked).detach().item()
            total_valid_words += torch.sum(padding_mask)

    total_epoch_valid_loss = total_epoch_valid_loss/total_valid_words
  
    print("Epoch -> ", epoch," Training Loss -> ", total_epoch_train_loss.item(), "Eval Loss -> ", total_epoch_valid_loss.item() )
  
    if min_val_loss > total_epoch_valid_loss:
        print("Writing Model at epoch ", epoch)
        torch.save(ictModel, './BestModel')
        min_val_loss = total_epoch_valid_loss
  

    scheduler.step(total_epoch_valid_loss.item())
```

The error is:

```bash
AttributeError                            Traceback (most recent call last)
/home/romainm/ml_project/main.py in line 11
      316 scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.8, patience=2, verbose = True)
     317 criterion = torch.nn.CrossEntropyLoss(reduction='none')
---> 318 min_val_loss = np.float('Inf')
     320 for epoch in tqdm(range(EPOCH)):
     321     total_epoch_train_loss = 0

File ~/miniconda3/envs/venv38/lib/python3.8/site-packages/numpy/__init__.py:305, in __getattr__(attr)
    300     warnings.warn(
    301         f"In the future `np.{attr}` will be defined as the "
    302         "corresponding NumPy scalar.", FutureWarning, stacklevel=2)
    304 if attr in __former_attrs__:
--> 305     raise AttributeError(__former_attrs__[attr])
    307 # Importing Tester requires importing all of UnitTest which is not a
    308 # cheap import Since it is mainly used in test suits, we lazy import it
    309 # here to save on the order of 10 ms of import time for most users
    310 #
    311 # The previous way Tester was imported also had a side effect of adding
    312 # the full `numpy.testing` namespace
    313 if attr == 'testing':

AttributeError: module 'numpy' has no attribute 'float'.
`np.float` was a deprecated alias for the builtin `float`. To avoid this error in existing code, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
```

Solution: `min_val_loss = float('Inf')`

### Low training

Switching to Adam optimizer with lr = 0.001

```python
EPOCH = 30

ictModel = ImageCaptionModel(16, 4, vocab_size, 512).to(device)
optimizer = torch.optim.Adam(ictModel.parameters(), lr = 0.00001)
# optimizer = torch.optim.Adam(ictModel.parameters(), lr = 0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.8, patience=2, verbose = True)
criterion = torch.nn.CrossEntropyLoss(reduction='none')
min_val_loss = float('Inf')
```

### During GPU training

```bash

4/30 [14:18<1:19:11, 182.77s/it]
/home/romainm/miniconda3/envs/venv38/lib/python3.8/site-packages/torch/nn/functional.py:5076: UserWarning: Support for mismatched key_padding_mask and attn_mask is deprecated. Use same type for both instead.
  warnings.warn(
Epoch ->  0  Training Loss ->  5.225950717926025 Eval Loss ->  7.342782974243164
Writing Model at epoch  0
Epoch ->  1  Training Loss ->  4.6445112228393555 Eval Loss ->  7.123194217681885
Writing Model at epoch  1
Epoch ->  2  Training Loss ->  4.43972635269165 Eval Loss ->  7.9219970703125
Epoch ->  3  Training Loss ->  4.28543758392334 Eval Loss ->  7.17313814163208
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/home/romainm/ml_project/main.py in line 41
     350     final_batch_loss.backward()
     351     optimizer.step()
---> 352     total_epoch_train_loss += torch.sum(loss_masked).detach().item()
     353     total_train_words += torch.sum(padding_mask)
     356 total_epoch_train_loss = total_epoch_train_loss/total_train_words

RuntimeError: CUDA error: unknown error
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

Reduce Batch Size: Reduce the batch size used during training and validation. A smaller batch size will consume less GPU memory. You can set smaller batch sizes in your data loaders or model training loop.

Use torch.no_grad() for Validation: Wrap your validation loop with torch.no_grad() to prevent unnecessary GPU memory consumption for gradient calculations during validation. This can be added as shown in the code below:

```python
with torch.no_grad():
    for caption_seq, target_seq, image_embed in valid_dataloader_resnet:
        # Your evaluation code here
```

Release GPU Memory: After each batch, release GPU memory by calling torch.cuda.empty_cache(). You can do this in your training loop after the optimizer step:

```python
final_batch_loss.backward()
optimizer.step()
total_epoch_train_loss += torch.sum(loss_masked).detach().item()
total_train_words += torch.sum(padding_mask)
torch.cuda.empty_cache()  # Release GPU memory
```

Check for Hardware Issues: Ensure your GPU is not experiencing hardware issues, such as overheating, that could lead to CUDA errors.

Check Compatibility: Verify that your PyTorch version is compatible with your CUDA version, and ensure that you have updated GPU drivers to avoid compatibility issues.

Hardware Availability: Ensure that the GPU you are using is available and not being used by other processes or users.

Here's an updated version of your code with the modifications:

```python
EPOCH = 30

ictModel = ImageCaptionModel(16, 4, vocab_size, 512).to(device)
optimizer = torch.optim.Adam(ictModel.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=2, verbose=True)
criterion = torch.nn.CrossEntropyLoss(reduction='none')
min_val_loss = float('Inf')

for epoch in tqdm(range(EPOCH)):
    total_epoch_train_loss = 0
    total_epoch_valid_loss = 0
    total_train_words = 0
    total_valid_words = 0
    ictModel.train()

    # Train Loop
    for caption_seq, target_seq, image_embed in train_dataloader_resnet:
        optimizer.zero_grad()

        image_embed = image_embed.squeeze(1).to(device)
        caption_seq = caption_seq.to(device)
        target_seq = target_seq.to(device)

        output, padding_mask = ictModel.forward(image_embed, caption_seq)
        output = output.permute(1, 2, 0)

        loss = criterion(output, target_seq)

        loss_masked = torch.mul(loss, padding_mask)

        final_batch_loss = torch.sum(loss_masked) / torch.sum(padding_mask)

        final_batch_loss.backward()
        optimizer.step()
        total_epoch_train_loss += torch.sum(loss_masked).detach().item()
        total_train_words += torch.sum(padding_mask)

        torch.cuda.empty_cache()  # Release GPU memory

    total_epoch_train_loss = total_epoch_train_loss / total_train_words

    # Evaluation Loop
    ictModel.eval()
    with torch.no_grad():
        for caption_seq, target_seq, image_embed in valid_dataloader_resnet:
            image_embed = image_embed.squeeze(1).to(device)
            caption_seq = caption_seq.to(device)
            target_seq = target_seq.to(device)

            output, padding_mask = ictModel.forward(image_embed, caption_seq)
            output = output.permute(1, 2, 0)

            loss = criterion(output, target_seq)

            loss_masked = torch.mul(loss, padding_mask)

            total_epoch_valid_loss += torch.sum(loss_masked).detach().item()
            total_valid_words += torch.sum(padding_mask)

    total_epoch_valid_loss = total_epoch_valid_loss / total_valid_words

    print("Epoch ->", epoch, "Training Loss ->", total_epoch_train_loss, "Eval Loss ->", total_epoch_valid_loss)

    if min_val_loss > total_epoch_valid_loss:
        print("Writing Model at epoch", epoch)
        torch.save(ictModel, './BestModel')
        min_val_loss = total_epoch_valid_loss

    scheduler.step(total_epoch_valid_loss)
```

**Solution**: Change learning rate to 0.00001