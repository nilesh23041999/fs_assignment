import torch
from feature_extractor import FeatureExtractor
import cv2
import torchvision.transforms as transforms

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = FeatureExtractor().to(device)


def similar_images(img_path,emb_path,count):
    
    emb = torch.load(emb_path)
    embeddings = emb["Features"]
    emb_indices = emb["image_index"]
    emb_path = emb["Image_path"]
   
    img = cv2.imread(img_path)
    raw = img
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    img = transforms.ToTensor()(img).unsqueeze(0).to(device)
    print(img.shape)
    with torch.no_grad():
        model.eval()
        x = model(img)  # feature_vector
    
    x = x.reshape(-1,x.shape[0])
    embeddings = embeddings.reshape((embeddings.shape[0], -1))

    # using cosine similarity for getting similar images
    cos = torch.nn.CosineSimilarity(dim=1)
    #print('check ',x.shape,embeddings.shape) 
    distance = cos(x, embeddings)
    #print(distance)
    
    keys = torch.argsort(distance)[1:count+1]
    return raw, keys,emb_indices,emb_path


