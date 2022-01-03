import torch
import timm
import vision_transformer as vits

model = vits.__dict__["vit_small"](patch_size=16, num_classes=0)


def getSupervisedModel():
    model_supervised = timm.create_model("deit_small_patch16_224", pretrained=True)
    return model_supervised


def getMidModel():
    midModel = torch.load(
        "../logs-scratch-local-10e/best_model.pth", map_location="cpu"
    ).backbone
    return midModel


def getFinalModel():
    url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
    state_dict = torch.hub.load_state_dict_from_url(
        url="https://dl.fbaipublicfiles.com/dino/" + url
    )
    model.load_state_dict(state_dict, strict=True)
    return model


allModels = {
    "Supervisado": getSupervisedModel(),
    "DINO - 10 épocas": getMidModel(),
    "DINO - 50 épocas": getFinalModel(),
}
