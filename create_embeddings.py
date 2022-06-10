import os
import torch
import torch.utils.data as DataLoader
from data_set import *
from img_transforms import *
from feature_extractor import *

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def custom_collate(batch):
    # each item in batch is (250, 150) as returned by __getitem__
    return torch.cat(batch, 0)


def my_collate(batch):
    c = torch.stack(batch, dim=1)
    return c

model = FeatureExtractor().to(device)

dataset = CustomDataset(imgs_path = r"E:\flix_stock\archive\visualsimilarity\bottoms_resized_png",transforms=transformations)
dataloader = DataLoader(dataset, batch_size=1)




x_random = torch.randn((1, 3, 224, 224)).to(device)

x = model(x_random) 

embeddings = torch.randn_like(x).to(device)
p = []
index_lst = []
img_path_lst = []

with torch.no_grad():
    model.eval()
    for index,(img,img_path) in enumerate(dataloader):
        img = img.to(device)
        feature_vectors = model(img).to(device)
        p.append(feature_vectors)
        index_lst.append(index)
        img_path_lst.append(img_path)
        

res = torch.stack(p)



# saving embeddings
torch.save({"image_index":index,"Image_path":img_path_lst,"Features":res}, os.path.join('E:\flix_stock\embedding.pt'))

