import pandas as pd
import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.text  import load_img, img_to_array


customers_df = pd.read_csv('/Users/miguel/Documents/h-and-m-personalized-fashion-recommendations/customers.csv')
articles_df = pd.read_csv('/Users/miguel/Documents/h-and-m-personalized-fashion-recommendations/articles.csv')
transactions_df = pd.read_csv('/Users/miguel/Documents/h-and-m-personalized-fashion-recommendations/transactions_train.csv')

image_paths = []

image_dir = '/Users/miguel/Documents/h-and-m-personalized-fashion-recommendations/images'

  
for rootdir, dirs, files in os.walk(image_dir):
       for file in files:
           if file.endswith('.jpg') or file.endswith('.png'):
              
               file_path = os.path.join(rootdir, file)
               try:
                   img = Image.open(file_path)
                   if img.mode == 'RGB':
                       image_paths.append(file_path)
               except IOError:
                   print(f"Could not read file: {file_path}")

print(f"Total images stored: {len(image_paths)}")

articles_df['article_id'] = articles_df['article_id'].astype(str)

 
image_labels_mapping = {}

for image_path in image_paths:
     
       article_id = os.path.basename(image_path).split('.')[0][1:] 
       product_type_name = articles_df[articles_df['article_id'] == article_id]['product_type_name']

       if not product_type_name.empty:
           image_labels_mapping[image_path] = product_type_name.values[0]

print(f"Total mapped images: {len(image_labels_mapping)}")

image_paths_list, labels_list = zip(*image_labels_mapping.items())

lb = LabelBinarizer()
labels_ohe = lb.fit_transform(np.array(labels_list))

loaded_images = []
for image_path in image_paths_list:
       image = load_img(image_path, target_size=(150, 150))
       image = img_to_array(image)
       image = image / 255.0  
       loaded_images.append(image)

np_images = np.array(loaded_images)

X_train, X_test, y_train, y_test = train_test_split(range(len(np_images)), test_size=0.20, stratify=labels_ohe)

print(f'Training dataset size: {len(X_train)}')
print(f'Testing dataset size: {len(X_test)}')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
