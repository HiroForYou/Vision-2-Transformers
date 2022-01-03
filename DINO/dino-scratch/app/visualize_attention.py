import torch.nn as nn
from torchvision import transforms as pth_transforms


def getAttentionMapOfFinalModel(model, device, img, i_head):
    patch_size = 16
    image_size = (480, 480)
    threshold = None

    img = img.convert("RGB")
    transform = pth_transforms.Compose(
        [
            pth_transforms.Resize(image_size),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    img = transform(img)
    # make the image divisible by the patch size
    w, h = (
        img.shape[1] - img.shape[1] % patch_size,
        img.shape[2] - img.shape[2] % patch_size,
    )
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    attentions = model.get_last_selfattention(img.to(device))

    nh = attentions.shape[1]  # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = (
        nn.functional.interpolate(
            attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest"
        )[0]
        .cpu()
        .detach()
        .numpy()
    )

    return attentions[i_head]
