import json
from PIL import Image
from io import BytesIO
import time
import os
import base64
import numpy as np
import sys
import os

import torch
from torchvision import transforms
from torch.autograd import Variable
import torchvision.utils as vutils
from pytorch_pretrained_vit import ViT, load_pretrained_weights

model_path = os.getcwd() +  "/../ViT/models"

def load_models():

    clfs = ["B_16_imagenet1k", "B_32_imagenet1k"]
    models = {}

    for clf_name in clfs:
        model = ViT(clf_name, pretrained=False)
        # the weights load from local
        
        load_pretrained_weights(model, weights_path=f'{model_path}/{clf_name}.pth') 
        model.eval()
        models[clf_name] = model

    return models


mapping_id_to_clfs = {
    0: "B_16_imagenet1k",
    1: "B_32_imagenet1k"
}

models = load_models()
print(f"models loaded ...")


def inference_ViT(data):

    image = data["image"]
    image = image[image.find(",")+1:]
    dec = base64.b64decode(image + "===")
    image = Image.open(BytesIO(dec))
    image = image.convert("RGB")


    # load the model with the selected classifier
    model_id = int(data["model_id"])
    clf = mapping_id_to_clfs[model_id]
    model = models[clf]

    # Preprocess image
    tfms = transforms.Compose([transforms.Resize(model.image_size),  # model.image_size = (384, 384)
                           transforms.ToTensor(), 
                           transforms.Normalize([0.5, 0.5, 0.5], 
                                                [0.5, 0.5, 0.5]),]
                            )

    image = tfms(image).unsqueeze(0)
    

    # Load class names
    labels_map = json.load(open(f'{model_path}/labels_map.txt'))
    labels_map = [labels_map[str(i)] for i in range(1000)]


    # Classify
    with torch.no_grad():
        outputs = model(image).squeeze(0)
    print('-----------------------------')

    output_predicts = []
    for idx in torch.topk(outputs, k=3).indices.tolist():
        prob = torch.softmax(outputs, -1)[idx].item()
        predict = '[{idx}] {label:<75} ({p:.2f}%)'.format(idx=idx, label=labels_map[idx], p=prob*100)
        print(predict)
        output_predicts.append(predict)

    return output_predicts
    
