from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Form, Response
from fastapi.middleware.cors import CORSMiddleware
import matplotlib.pyplot as plt
from PIL import Image
import timm
import torch
import torch.nn.functional as F
from torchvision import transforms
import uvicorn

import vision_transformer as vits
from visualize_attention import getAttentionMapOfFinalModel

device = torch.device("cpu")
model = vits.__dict__["vit_small"](patch_size=16, num_classes=0)

def getSupervisedModel():
    model_supervised = timm.create_model("deit_small_patch16_224", pretrained=True)
    return model_supervised

def getMidModel():
    midModel = torch.load(
        "DINO/dino-scratch/logs-scratch-local-10e/best_model.pth", map_location="cpu"
    ).backbone
    return midModel

def getFinalModel():
    url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
    state_dict = torch.hub.load_state_dict_from_url(
        url="https://dl.fbaipublicfiles.com/dino/" + url
    )
    model.load_state_dict(state_dict, strict=True)
    return model


models = {
    "Supervisado": getSupervisedModel(),
    "DINO - 10 épocas": getMidModel(),
    "DINO - 50 épocas": getFinalModel(),
}

app = FastAPI(
    title="Solver Captcha", description="Endpoint Solver Captcha", version="0.0.1"
)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_last_attention(backbone, x):
    """Get the attention weights of CLS from the last self-attention layer.

    Parameters
    ----------
    backbone : timm.models.vision_transformer.VisionTransformer
        Instantiated Vision Transformer. Note that we will in-place
        take the `head` attribute and replace it with `nn.Identity`.

    x : torch.Tensor
        Batch of images of shape `(n_samples, 3, size, size)`.

    Returns
    -------
    torch.Tensor
        Attention weights `(n_samples, n_heads, n_patches)`.
    """
    attn_module = backbone.blocks[-1].attn
    n_heads = attn_module.num_heads

    # define hook
    inp = None

    def fprehook(self, inputs):
        nonlocal inp
        inp = inputs[0]

    # Register a hook
    handle = attn_module.register_forward_pre_hook(fprehook)

    # Run forward pass
    _ = backbone(x)
    handle.remove()

    B, N, C = inp.shape
    qkv = (
        attn_module.qkv(inp)
        .reshape(B, N, 3, n_heads, C // n_heads)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, _ = qkv[0], qkv[1], qkv[2]

    attn = (q @ k.transpose(-2, -1)) * attn_module.scale
    attn = attn.softmax(dim=-1)

    return attn[:, :, 0, 1:]



def threshold(attn, k=30):
    n_heads = len(attn)
    indices = attn.argsort(dim=1, descending=True)[:, k:]

    for head in range(n_heads):
        attn[head, indices[head]] = 0

    attn /= attn.sum(dim=1, keepdim=True)

    return attn


def visualize_attention(img, backbone, k=30):
    """Create attention image.

    Parameteres
    -----------
    img : PIL.Image
        RGB image.

    backbone : timm.models.vision_transformer.VisionTransformer
        The vision transformer.

    Returns
    -------
    new_img : torch.Tensor
        Image of shape (n_heads, 1, height, width).
    """
    # imply parameters

    patch_size = backbone.patch_embed.proj.kernel_size[0]

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    device = next(backbone.parameters()).device
    x = transform(img)[None, ...].to(device)
    attn = get_last_attention(backbone, x)[0]  # (n_heads, n_patches)
    attn = attn / attn.sum(dim=1, keepdim=True)  # (n_heads, n_patches)
    attn = threshold(attn, k)
    attn = attn.reshape(-1, 14, 14)  # (n_heads, 14, 14)
    attn = F.interpolate(attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0]

    return attn


def run_predict(model_name, i_head, image_file):

    image = Image.open(BytesIO(image_file.read())).convert("RGB")

    if model_name in ["Supervisado", "DINO - 10 épocas"]:
        attns = (
            visualize_attention(image, models[model_name], k=30)
            .detach()[:]
            .permute(1, 2, 0)
            .numpy()
        )

        kwargs = {"vmin": 0, "vmax": 0.24}
        attn = attns[..., i_head]
        fig = plt.figure(figsize=(4, 4), dpi=43.0)
        fig.figimage(attn, **kwargs)
        img_buf = BytesIO()
        fig.savefig(img_buf, format="png")
        plt.close()
        return img_buf
    else:
        attn = getAttentionMapOfFinalModel(models[model_name], device, image, i_head)
        fig = plt.figure(dpi=70.0)
        fig.figimage(attn)
        img_buf = BytesIO()
        fig.savefig(img_buf, format="png")
        plt.close()
        return img_buf


@app.get("/")
def home():
    return {"message": "Endpoint DINO"}


@app.post("/getAttentionMap")
def _file_upload(image_file: UploadFile = File(...), model_name: str = Form(...), i_head: str = Form(...)):
    try:
        attn = run_predict(model_name, int(i_head), image_file.file)
        return Response(content=attn.getvalue(), media_type="image/png")

    except Exception as e:
        return {"error": f"{e}, actualice el despliegue"}


if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True, workers=1)