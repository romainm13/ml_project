# %%
# Remove images in Flicker8k_Dataset that are not in Flickr8k.token.txt
# Sometimes FileNotFoundError: [Errno 2] No such file or directory: 'Flicker8k_Dataset/2258277193_586949ec62.jpg'

""" ERROR HAPPENS HERE
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
"""

#%%
# Get all images' names in Flickr8k.token.txt
import pandas as pd
df = pd.read_csv("Flickr8k_text/Flickr8k.token.txt", sep='\t', header=None)
df.rename(columns={0:'image', 1:'caption'}, inplace=True)
display(df.head())
print("Length of df: ", len(df))

# Remove #digit from image name
df['image'] = df['image'].apply(lambda x : x.split('.jpg')[0] + '.jpg')
display(df.head())
print("Number of unique images: ", len(df['image'].unique()))

#%%
# Get all images' names in Flicker8k_Dataset
import glob
import os

img_names = glob.glob("Flicker8k_Dataset/*.jpg")
print("Number of images in Flicker8k_Dataset: ", len(img_names))

img_names = [os.path.basename(img_name) for img_name in img_names]
for i in range(5):
    print(img_names[i])

# Remove lines in df that are not in Flicker8k_Dataset
print("Length of df: ", len(df))
df = df[df['image'].isin(img_names)]
print("Length of df: ", len(df))