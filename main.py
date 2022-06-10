import os
from similarity_search import similar_images
import argparse
import cv2 as cv2
import matplotlib.pyplot as plt
import torch


device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str , help="path of input image")
    parser.add_argument('--count', type=int, default=10, help="number of similar images to be querried from database" )
    parser.add_argument('--embedding_path',type= str,help="path of embeddings")
    
    arg = parser.parse_args()
    return arg


### For plotting 10 similar images
def visualize(raw,indices,emb_path):
    window_name = 'image'
    image_lst = []
    indices = indices.tolist()
    #print(":::::::::::",emb_path[indices[0]][0])
    for i in range(len(indices)):
        image_lst.append(cv2.imread(emb_path[indices[i]][0]))
    
    
        
    f, axarr = plt.subplots(2,5)
    axarr[0,0].imshow(image_lst[0])
    axarr[0,1].imshow(image_lst[1])
    axarr[0,2].imshow(image_lst[2])
    axarr[0,3].imshow(image_lst[3])
    axarr[0,4].imshow(image_lst[4])
    axarr[1,0].imshow(image_lst[5])
    axarr[1,1].imshow(image_lst[6])
    axarr[1,2].imshow(image_lst[7])
    axarr[1,3].imshow(image_lst[8])
    axarr[1,4].imshow(image_lst[9])
    
    plt.show()
    cv2.imshow(window_name,raw)
    cv2.waitKey(0) 

if __name__ == "__main__":
    arg = args()
    embedding_path = arg.embedding_path
    img_path = arg.input
    count = arg.count
    raw, indices,emb_indices,emb_path = similar_images(img_path = img_path , emb_path=embedding_path,count = count)
    visualize(raw,indices,emb_path)
    

    
    
    