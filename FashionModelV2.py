import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image

  
csv_path = '/Users/miguel/Documents/h-and-m-personalized-fashion-recommendations/articles.csv'
image_dir = '/Users/miguel/Documents/h-and-m-personalized-fashion-recommendations/images'
 
articles_df = pd.read_csv(csv_path)
  
class ClothingDataset(Dataset):
      
       def __init__(self, image_paths, labels, transform=None):

           self.image_paths = image_paths
           self.labels = labels
           self.transform = transform

       def __getitem__(self, item):
           path, target = self.image_paths[item], self.labels[item]
           img = Image.open(path).convert('RGB')
           if self.transform is not None:
               img = self.transform(img)
           return img, target

       def __len__(self):
           return len(self.labels)
       
class ConvNet(nn.Module):
       pass

 
def create_image_labels_mapping(image_paths, articles_df):
       image_labels_mapping = {}
       for image_path in image_paths:
           filename = os.path.basename(image_path)
           article_id = filename.lstrip('0').rstrip('.jpg')
           product_type_name = articles_df[articles_df['article_id'] == int(article_id)]['product_type_name']
           if not product_type_name.empty:
               image_labels_mapping[image_path] = product_type_name.values[0]

       return image_labels_mapping

       if not_found != 0:
           print(f"{not_found} article IDs not found in dataframe.")
       print(f"Total images: {len(image_paths)}")
       print(f"Total image-label mappings: {len(image_labels_mapping)}")

       if len(image_labels_mapping) == 0:
           print("Image-Label mapping failed. Check your article IDs and dataframe.")

       return image_labels_mapping


 
file_paths = []
for subdir, dirs, files in os.walk(image_dir):
       for filename in files:
           filepath = subdir + os.sep + filename
           if filepath.endswith(".jpg") or filepath.endswith(".png"):
               file_paths.append(filepath)


def debug_unfound_article_ids(image_paths, articles_df):
       for image_path in image_paths:
           article_id = os.path.basename(image_path).split('.')[0]

           if article_id not in articles_df['article_id'].values:
               print(f"Article ID not found in dataframe: {article_id}")

debug_unfound_article_ids(file_paths, articles_df)


articles_df = articles_df.sort_values(by='article_id')

  
file_paths.sort(key=lambda x: os.path.basename(x).split('.')[0])

if 'article_id' in articles_df.columns:
       print("First five rows in articles_df:")
       print(articles_df.head())

   
print("First 5 filenames:")
for path in file_paths[:5]:
       print(os.path.basename(path))

 
image_labels_mapping = create_image_labels_mapping(file_paths, articles_df)
labels = list(image_labels_mapping.values())
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(file_paths, labels_encoded, test_size=0.20, stratify=labels_encoded)

   
transform = transforms.Compose([
       transforms.Resize((150, 150)),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
      ]
   )

  
train_data = ClothingDataset(X_train, y_train, transform)
test_data = ClothingDataset(X_test, y_test, transform)

  
batch_size = 32
num_workers = 0

train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

  
model = ConvNet()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()


def train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=25):

       device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       model = model.to(device)

       for epoch in range(num_epochs):
           print(f'Epoch {epoch+1}/{num_epochs}')
           print('-' * 10)

           for phase in ['train', 'val']:
               if phase == 'train':
                   data_loader = train_loader
                   model.train()  
                   data_loader = test_loader
                   model.eval()   

               running_loss = 0.0
               running_corrects = 0

               for inputs, labels in data_loader:
                   inputs = inputs.to(device)
                   labels = labels.to(device)

                 
                   optimizer.zero_grad()

                  
                   with torch.set_grad_enabled(phase == 'train'):
                       outputs = model(inputs)
                       _, preds = torch.max(outputs, 1)
                       loss = criterion(outputs, labels)

                     
                       if phase == 'train':
                           loss.backward()
                           optimizer.step()

                  
                   running_loss += loss.item() * inputs.size(0)
                   running_corrects += torch.sum(preds == labels.data)

               if phase == 'train':
                   dataset_size = len(train_data)
               else:
                   dataset_size = len(test_data)

               epoch_loss = running_loss / dataset_size
               epoch_acc = running_corrects.double() / dataset_size

               print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

       return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = train_model(model, criterion, optimizer, num_epochs=25)