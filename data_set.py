import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import os

class CustomDataset(Dataset):
    def __init__(self,imgs_path,transforms=None):
        self.imgs_path = imgs_path
        self.file_list = os.listdir(imgs_path)
        self.transforms = transforms 
        self.img_dim = (224, 224)
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        #print(self.file_list)
        img_path = os.path.join(self.imgs_path, self.file_list[index])
        img = cv2.imread(img_path)
        
        if self.transforms is not None:
            #img = cv2.resize(img, (150, 150))
            img = cv2.imread(img_path)
            
            img  = self.transforms(img)
        
        return img,img_path
     
    
    
