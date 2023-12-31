
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
from IPython.display import display
import nltk
from nltk.translate.bleu_score import sentence_bleu

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
    
# %% 
a_file = open("./EncodedImageTrainResNet.pkl", "wb")
pickle.dump(extract_imgFtr_ResNet_train, a_file)
a_file.close()

# %%
extract_imgFtr_ResNet_valid = {}
for image_name, t_img in tqdm(valid_ImageDataloader_ResNet):
    t_img = t_img.to(device)
    embdg = get_vector(t_img)
 
    extract_imgFtr_ResNet_valid[image_name[0]] = embdg

# %%
a_file = open("./EncodedImageValidResNet.pkl", "wb")
pickle.dump(extract_imgFtr_ResNet_valid, a_file)
a_file.close()

# %%
# ## Create DataLoader which will be used to load data into Transformer Model.
# ## FlickerDataSetResnet will return caption sequence, 1 timestep left shifted caption sequence which model will predict and Stored Image features from ResNet.

class FlickerDataSetResnet():
    def __init__(self, data, pkl_file):
        self.data = data
        self.encodedImgs = pd.read_pickle(pkl_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        real_capt = self.data.iloc[idx]['caption']
        caption_seq = self.data.iloc[idx]['text_seq']
        target_seq = caption_seq[1:]+[0]

        image_name = self.data.iloc[idx]['image']
        image_tensor = self.encodedImgs[image_name]
        image_tensor = image_tensor.permute(0,2,3,1)
        image_tensor_view = image_tensor.view(image_tensor.size(0), -1, image_tensor.size(3))

        return torch.tensor(caption_seq), torch.tensor(target_seq), image_tensor_view, real_capt

train_dataset_resnet = FlickerDataSetResnet(train, 'EncodedImageTrainResNet.pkl')
train_dataloader_resnet = DataLoader(train_dataset_resnet, batch_size = 32, shuffle=True)

valid_dataset_resnet = FlickerDataSetResnet(valid, 'EncodedImageValidResNet.pkl')
valid_dataloader_resnet = DataLoader(valid_dataset_resnet, batch_size = 32, shuffle=True)

# %% 
# ## Create Transformer Decoder Model. This Model will take caption sequence and the extracted resnet image features as input and ouput 1 timestep shifted (left) caption sequence. 
# ## In the Transformer decoder, lookAhead and padding mask has also been applied

# %%
# ### Position Embedding

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        

    def forward(self, x):
        if self.pe.size(0) < x.size(0):
            self.pe = self.pe.repeat(x.size(0), 1, 1).to(device)
        self.pe = self.pe[:x.size(0), : , : ]
        
        x = x + self.pe
        return self.dropout(x)

# %%
# ## Transformer Decoder

class ImageCaptionModel(nn.Module):
    def __init__(self, n_head, n_decoder_layer, vocab_size, embedding_size):
        super(ImageCaptionModel, self).__init__()
        self.pos_encoder = PositionalEncoding(embedding_size, 0.1)
        self.TransformerDecoderLayer = nn.TransformerDecoderLayer(d_model =  embedding_size, nhead = n_head)
        self.TransformerDecoder = nn.TransformerDecoder(decoder_layer = self.TransformerDecoderLayer, num_layers = n_decoder_layer)
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size , embedding_size)
        self.last_linear_layer = nn.Linear(embedding_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.last_linear_layer.bias.data.zero_()
        self.last_linear_layer.weight.data.uniform_(-initrange, initrange)

    def generate_Mask(self, size, decoder_inp):
        decoder_input_mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        decoder_input_mask = decoder_input_mask.float().masked_fill(decoder_input_mask == 0, float('-inf')).masked_fill(decoder_input_mask == 1, float(0.0))

        decoder_input_pad_mask = decoder_inp.float().masked_fill(decoder_inp == 0, float(0.0)).masked_fill(decoder_inp > 0, float(1.0))
        decoder_input_pad_mask_bool = decoder_inp == 0

        return decoder_input_mask, decoder_input_pad_mask, decoder_input_pad_mask_bool

    def forward(self, encoded_image, decoder_inp):
        encoded_image = encoded_image.permute(1,0,2)
        

        decoder_inp_embed = self.embedding(decoder_inp)* math.sqrt(self.embedding_size)
        
        decoder_inp_embed = self.pos_encoder(decoder_inp_embed)
        decoder_inp_embed = decoder_inp_embed.permute(1,0,2)
        

        decoder_input_mask, decoder_input_pad_mask, decoder_input_pad_mask_bool = self.generate_Mask(decoder_inp.size(1), decoder_inp)
        decoder_input_mask = decoder_input_mask.to(device)
        decoder_input_pad_mask = decoder_input_pad_mask.to(device)
        decoder_input_pad_mask_bool = decoder_input_pad_mask_bool.to(device)
        

        decoder_output = self.TransformerDecoder(tgt = decoder_inp_embed, memory = encoded_image, tgt_mask = decoder_input_mask, tgt_key_padding_mask = decoder_input_pad_mask_bool)
        
        final_output = self.last_linear_layer(decoder_output)

        return final_output,  decoder_input_pad_mask

# %%
# ##  Train the Model
# ### The cross entropy loss has been masked at time steps where input token is <'pad'>.

train_losses = []
val_losses = []
# val_accuracy = []

start_token = word_to_index['<start>']
end_token = word_to_index['<end>']
pad_token = word_to_index['<pad>']
max_seq_len = 33

EPOCH = 30

ictModel = ImageCaptionModel(16, 4, vocab_size, 512).to(device)
optimizer = torch.optim.Adam(ictModel.parameters(), lr = 0.00001)
# optimizer = torch.optim.Adam(ictModel.parameters(), lr = 0.001) NOT WORKING AT ALL BUG
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.8, patience=2, verbose = True)
criterion = torch.nn.CrossEntropyLoss(reduction='none')
min_val_loss = float('Inf')

for epoch in tqdm(range(EPOCH)):
    total_epoch_train_loss = 0
    total_epoch_valid_loss = 0
    total_train_words = 0
    total_valid_words = 0
    # total_accuracy_train = 0
    # total_accuracy_valid = 0

    ### Train Loop
    ictModel.train()
    for caption_seq, target_seq, image_embed, real_capt in train_dataloader_resnet:
        optimizer.zero_grad()

        image_embed = image_embed.squeeze(1).to(device)
        caption_seq = caption_seq.to(device)
        target_seq = target_seq.to(device)

        """
        image_embed.shape torch.Size([32, 49, 512])
        caption_seq.shape torch.Size([32, 33])
        """
        output, padding_mask = ictModel.forward(image_embed, caption_seq)
        output = output.permute(1, 2, 0)
        """
        output.shape torch.Size([32, 8360, 33])
        """

        loss = criterion(output,target_seq)

        loss_masked = torch.mul(loss, padding_mask)

        final_batch_loss = torch.sum(loss_masked)/torch.sum(padding_mask)

        final_batch_loss.backward()
        optimizer.step()
        total_epoch_train_loss += torch.sum(loss_masked).detach().item()
        total_train_words += torch.sum(padding_mask)

        ####
        # print("## len real_capt", len(real_capt))
        # for img_emb,ref_cap in zip(image_embed, real_capt):
        #     print("compteur", compteur)
        #     start = time.time()
        #     predicted_sentence = []
            
        #     img_emb = img_emb.unsqueeze(0)
        #     """
        #     torch.Size([1, 49, 512])
        #     """
            
        #     input_seq = [pad_token]*max_seq_len
        #     input_seq[0] = start_token

        #     input_seq = torch.tensor(input_seq).unsqueeze(0).to(device)
            
        #     with torch.no_grad():
        #         for eval_iter in range(0, max_seq_len-1):
                    
        #             output, padding_mask = ictModel.forward(img_emb, input_seq)
        #             output = output[eval_iter, 0, :]
        #             # print("output.shape", output.shape)

        #             values = torch.topk(output, 1).values.tolist()
        #             indices = torch.topk(output, 1).indices.tolist()

        #             next_word_index = random.choices(indices, values, k = 1)[0]
        #             next_word = index_to_word[next_word_index]

        #             # print("input_seq.shape", input_seq.shape)
        #             input_seq[:, eval_iter+1] = next_word_index

        #             if next_word == '<end>' :
        #                 break
                    
        #             predicted_sentence.append(next_word)
            
        #     total_accuracy_train += sentence_bleu(ref_cap,predicted_sentence)
        #     print("time", time.time() - start)
        #     compteur += 1

        # total_accuracy_train = total_accuracy_train/len(real_capt)
 
    total_epoch_train_loss = total_epoch_train_loss/total_train_words
    
    ### Eval Loop
    ictModel.eval()
    with torch.no_grad():
        for caption_seq, target_seq, image_embed, real_capt in valid_dataloader_resnet:
            image_embed = image_embed.squeeze(1).to(device)
            caption_seq = caption_seq.to(device)
            target_seq = target_seq.to(device)

            output, padding_mask = ictModel.forward(image_embed, caption_seq)
            output = output.permute(1, 2, 0)

            loss = criterion(output,target_seq)

            loss_masked = torch.mul(loss, padding_mask)

            total_epoch_valid_loss += torch.sum(loss_masked).detach().item()
            total_valid_words += torch.sum(padding_mask)

            ####
            # for img_emb, ref_cap in zip(image_embed, real_capt):
            #     predicted_sentence = []
                
            #     img_emb = img_emb.unsqueeze(0)
                 
            #     input_seq = [pad_token]*max_seq_len
            #     input_seq[0] = start_token

            #     input_seq = torch.tensor(input_seq).unsqueeze(0).to(device)
            
            #     for eval_iter in range(0, max_seq_len-1):

            #         output, padding_mask = ictModel.forward(img_emb, input_seq)
            #         output = output[eval_iter, 0, :]

            #         values = torch.topk(output, 1).values.tolist()
            #         indices = torch.topk(output, 1).indices.tolist()

            #         next_word_index = random.choices(indices, values, k = 1)[0]
            #         next_word = index_to_word[next_word_index]

            #         input_seq[:, eval_iter+1] = next_word_index

            #         if next_word == '<end>' :
            #             break

            #     predicted_sentence.append(next_word)
            
            # sentence_bleu_score = sentence_bleu(ref_cap, predicted_sentence)
            # total_accuracy_valid += sentence_bleu_score
        
        # total_accuracy_valid = total_accuracy_valid/len(real_capt)
        # compteur_valid_loader += 1
        # print(f"{compteur_valid_loader}/{len(valid_dataloader_resnet)}")

    total_epoch_valid_loss = total_epoch_valid_loss/total_valid_words
  
    print("Epoch -> ", epoch," Training Loss -> ", total_epoch_train_loss.item(), "Eval Loss -> ", total_epoch_valid_loss.item() )
  
    if min_val_loss > total_epoch_valid_loss:
        print("Writing Model at epoch ", epoch)
        torch.save(ictModel, './BestModel')
        min_val_loss = total_epoch_valid_loss

    scheduler.step(total_epoch_valid_loss.item())

    train_losses.append(total_epoch_train_loss.item())
    val_losses.append(total_epoch_valid_loss.item())
    
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and validation Loss Over Epochs')
plt.legend()
plt.show()

# plt.figure(figsize=(10, 5))
# plt.plot(total_accuracy_valid, label='Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Validation Accuracy Over Epochs')
# plt.legend()
# plt.show()


# %% ## Lets Generate Captions !!!

model = torch.load('./BestModel')
start_token = word_to_index['<start>']
end_token = word_to_index['<end>']
pad_token = word_to_index['<pad>']
max_seq_len = 33
print(start_token, end_token, pad_token)

valid_img_embed = pd.read_pickle('EncodedImageValidResNet.pkl')

# %% ### Here in the below function,we are generating caption in beam search. K defines the topK token to look at each time step

def generate_caption(K, img_nm): 
    img_loc = '/home/romainm/ml_project/Flicker8k_Dataset/'+str(img_nm)
    image = Image.open(img_loc).convert("RGB")
    plt.imshow(image)

    model.eval() 
    valid_img_df = valid[valid['image']==img_nm]
    print("Actual Caption : ")
    print(valid_img_df['caption'].tolist())
    img_embed = valid_img_embed[img_nm].to(device)

    img_embed = img_embed.permute(0,2,3,1)
    img_embed = img_embed.view(img_embed.size(0), -1, img_embed.size(3))

    input_seq = [pad_token]*max_seq_len
    input_seq[0] = start_token

    input_seq = torch.tensor(input_seq).unsqueeze(0).to(device)
    predicted_sentence = []
    with torch.no_grad():
        for eval_iter in range(0, max_seq_len):

            output, padding_mask = model.forward(img_embed, input_seq)

            output = output[eval_iter, 0, :]

            values = torch.topk(output, K).values.tolist()
            indices = torch.topk(output, K).indices.tolist()

            next_word_index = random.choices(indices, values, k = 1)[0]

            next_word = index_to_word[next_word_index]

            input_seq[:, eval_iter+1] = next_word_index


            if next_word == '<end>' :
                break

            predicted_sentence.append(next_word)
    print("\n")
    print("Predicted caption : ")
    print(" ".join(predicted_sentence+['.']))

#%% ### 1st Example 
generate_caption(1, unq_valid_imgs.iloc[50]['image'])

# %% [code]
generate_caption(2, unq_valid_imgs.iloc[50]['image'])

# %% ### 2nd Example
generate_caption(1, unq_valid_imgs.iloc[100]['image'])

# %% [code]
generate_caption(2, unq_valid_imgs.iloc[100]['image'])

# %% [### 3rd Example
generate_caption(1, unq_valid_imgs.iloc[500]['image'])

# %% [code]
generate_caption(2, unq_valid_imgs.iloc[500]['image'])

# %% ### 4rth Example
generate_caption(1, unq_valid_imgs.iloc[600]['image'])

# %% [code]
generate_caption(2, unq_valid_imgs.iloc[600]['image'])

# %%
