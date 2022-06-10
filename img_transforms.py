import torchvision.transforms as transforms


transformations = transforms.Compose([transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
]) 