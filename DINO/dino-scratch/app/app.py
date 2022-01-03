import io
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
import timm
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch.nn.functional as F

import vision_transformer as vits
from visualize_attention import getAttentionMapOfFinalModel

device = torch.device("cpu")
# build model
model = vits.__dict__["vit_small"](patch_size=16, num_classes=0)
url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
state_dict = torch.hub.load_state_dict_from_url(
    url="https://dl.fbaipublicfiles.com/dino/" + url
)
model.load_state_dict(state_dict, strict=True)


models = {
    #"Supervisado": timm.create_model("deit_small_patch16_224", pretrained=True),
    "DINO - 10 épocas": torch.load(
        "DINO/dino-scratch/logs-scratch-local-10e/best_model.pth", map_location="cpu"
    ).backbone,
}

dataset = ImageFolder("DINO/dino-scratch/data_deploy/")


st.title("Demo interactiva de DINO")

model_name = st.sidebar.selectbox(
    #"Seleccione el modelo", ("Supervisado", "DINO - 10 épocas", "DINO - 50 épocas")
    "Seleccione el modelo", ("DINO - 10 épocas", "DINO - 50 épocas")
)
i_image = st.sidebar.slider(
    "Seleccione la imagen", min_value=0, max_value=len(dataset) - 1, value=18
)
i_head = st.sidebar.slider(
    "Seleccione el número de head", min_value=0, max_value=5, value=3
)


st.write(f"## Usando modelo {model_name}")

footer = """<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: #6A98F0; 
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Desarrollado con ❤️ por <a style='display: block; text-align: center;' href="https://www.cristhianwiki.com/" target="_blank">Cristhian Wiki</a></p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)


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


def plotAttention():
    img = dataset[i_image][0]  # 1949
    col1, col2 = st.columns(2)
    tform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
        ]
    )
    if model_name in models.keys():
        attns = (
            visualize_attention(img, models[model_name], k=30)
            .detach()[:]
            .permute(1, 2, 0)
            .numpy()
        )

        # original image
        plt.imshow(tform(img))
        plt.axis("off")
        plt.tight_layout()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png")
        original = Image.open(img_buf)
        col1.header("Original")
        col1.image(original, use_column_width=True)

        kwargs = {"vmin": 0, "vmax": 0.24}
        plt.imshow(attns[..., i_head], **kwargs)
        plt.axis("off")
        plt.tight_layout()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png")
        map_attention = Image.open(img_buf)
        col2.header(f"Head Attention {i_head}")
        col2.image(map_attention, use_column_width=True)

    else:
        attn = getAttentionMapOfFinalModel(model, device, img, i_head)

        # original image
        plt.imshow(tform(img))
        plt.axis("off")
        plt.tight_layout()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png")
        original = Image.open(img_buf)
        col1.header("Original")
        col1.image(original, use_column_width=True)

        plt.imshow(attn)
        plt.axis("off")
        plt.tight_layout()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png")
        map_attention = Image.open(img_buf)
        col2.header(f"Head Attention {i_head}")
        col2.image(map_attention, use_column_width=True)


plotAttention()
