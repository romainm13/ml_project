
# %%
import os
import glob

import pandas as pd
from collections import Counter 
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
import pickle
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

pd.set_option('display.max_colwidth', None)
print(torch.__version__)

#%%
"""
1000268201_693b08cb0e.jpg#0	A child in a pink dress is climbing up a set of stairs in an entry way .
1000268201_693b08cb0e.jpg#1	A girl going into a wooden building .
1000268201_693b08cb0e.jpg#2	A little girl climbing into a wooden playhouse .
1000268201_693b08cb0e.jpg#3	A little girl climbing the stairs to her playhouse .
1000268201_693b08cb0e.jpg#4	A little girl in a pink dress going into a wooden cabin .
1001773457_577c3a7d70.jpg#0	A black dog and a spotted dog are fighting
1001773457_577c3a7d70.jpg#1	A black dog and a tri-colored dog playing with 
"""
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

#%% Preprocessing -> Remove Single Character and non alpha Words. 
# Add , and tokens. token is appended such that length in max_seq_len 
# (maximum length across all captions which is 33 in our case)

def remove_single_char_word(word_list):
    lst = []
    for word in word_list:
        if len(word)>1:
            lst.append(word)

    return lst

df['cleaned_caption'] = df['caption'].apply(lambda caption : ['<start>'] + [word.lower() if word.isalpha() else '' for word in caption.split(" ")] + ['<end>'])
df['cleaned_caption']  = df['cleaned_caption'].apply(lambda x : remove_single_char_word(x))

df['seq_len'] = df['cleaned_caption'].apply(lambda x : len(x))
max_seq_len = df['seq_len'].max()
print(max_seq_len)

df.drop(['seq_len'], axis = 1, inplace = True)
df['cleaned_caption'] = df['cleaned_caption'].apply(lambda caption : caption + ['<pad>']*(max_seq_len-len(caption)) )

display(df.head(2))

# %% ## Create Vocab and mapping of token to ID

word_list = df['cleaned_caption'].apply(lambda x : " ".join(x)).str.cat(sep = ' ').split(' ')
word_dict = Counter(word_list)
word_dict =  sorted(word_dict, key=word_dict.get, reverse=True)

print(len(word_dict))
print(word_dict[:5])

# ### Vocab size is 8360

vocab_size = len(word_dict)
print(vocab_size)

index_to_word = {index: word for index, word in enumerate(word_dict)}
word_to_index = {word: index for index, word in enumerate(word_dict)}
print(len(index_to_word), len(word_to_index))

# %% ### Covert sequence of tokens to IDs
df['text_seq']  = df['cleaned_caption'].apply(lambda caption : [word_to_index[word] for word in caption] )
display(df.head(2))

# %% ## Split In Train and validation data. Same Image should not be present in both training and validation data 
df = df.sort_values(by = 'image')
train = df.iloc[:int(0.9*len(df))]
valid = df.iloc[int(0.9*len(df)):]

print(len(train), train['image'].nunique())
print(len(valid), valid['image'].nunique())


# %% ## Extract features from Images Using Resnet
train_samples = len(train)
print(train_samples)

unq_train_imgs = train[['image']].drop_duplicates()
unq_valid_imgs = valid[['image']].drop_duplicates()
print(len(unq_train_imgs), len(unq_valid_imgs))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

class extractImageFeatureResNetDataSet():
    def __init__(self, data):
        self.data = data 
        self.scaler = transforms.Resize([224, 224])
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()
    def __len__(self):  
        return len(self.data)

    def __getitem__(self, idx):

        image_name = self.data.iloc[idx]['image']
        img_loc = 'Flicker8k_Dataset/'+ str(image_name)

        img = Image.open(img_loc)
        t_img = self.normalize(self.to_tensor(self.scaler(img)))

        return image_name, t_img

#%%
train_ImageDataset_ResNet = extractImageFeatureResNetDataSet(unq_train_imgs)
train_ImageDataloader_ResNet = DataLoader(train_ImageDataset_ResNet, batch_size = 1, shuffle=False)

valid_ImageDataset_ResNet = extractImageFeatureResNetDataSet(unq_valid_imgs)
valid_ImageDataloader_ResNet = DataLoader(valid_ImageDataset_ResNet, batch_size = 1, shuffle=False)

resnet18 = torchvision.models.resnet18(pretrained=True).to(device)
resnet18.eval()
list(resnet18._modules)

resNet18Layer4 = resnet18._modules.get('layer4').to(device)

# %% 
def get_vector(t_img):
    t_img = Variable(t_img)
    my_embedding = torch.zeros(1, 512, 7, 7)
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    
    h = resNet18Layer4.register_forward_hook(copy_data)
    resnet18(t_img)
    
    h.remove()
    return my_embedding

extract_imgFtr_ResNet_train = {}
for image_name, t_img in tqdm(train_ImageDataloader_ResNet):
    t_img = t_img.to(device)
    embdg = get_vector(t_img)
    
    extract_imgFtr_ResNet_train[image_name[0]] = embdg
