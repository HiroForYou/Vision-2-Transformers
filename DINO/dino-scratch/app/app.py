import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
from torchvision.datasets import ImageFolder
import requests
from PIL import Image
from io import BytesIO
from torchvision import transforms

from config import URL

st.set_page_config(
    page_title="demo DINO",
    page_icon="fire",
    layout="wide",
)


def getPrediction(image, model_name, i_head):
    byte_io = BytesIO()
    image.save(byte_io, "png")
    byte_io.seek(0)
    response = requests.post(
        URL,
        files={"image_file": ("1.png", byte_io, "image/png")},
        data={"model_name": model_name, "i_head": i_head},
    )
    attn = Image.open(BytesIO(response.content))
    return attn


@st.cache(ttl=0.4 * 3600)
def getDataset():
    return ImageFolder("DINO/dino-scratch/data_deploy/")


dataset = getDataset()

st.title("Demo interactiva de DINO")

model_name = st.sidebar.selectbox(
    "Seleccione el modelo",
    ("Supervisado", "DINO - 10 épocas", "DINO - 50 épocas")
    # "Seleccione el modelo", ("DINO - 10 épocas", "DINO - 50 épocas")
)
i_image = st.sidebar.slider(
    "Seleccione la imagen", min_value=0, max_value=len(dataset) - 1, value=18
)
i_head = st.sidebar.slider(
    "Seleccione el número de head", min_value=0, max_value=5, value=3
)


st.write(f"#### Usando modelo {model_name}")

image_file = st.file_uploader("Subir imagen", type=["png", "jpg", "jpeg"])


def plotAttention():
    tform = transforms.Compose([transforms.Resize((224, 224))])
    if image_file is not None:
        img = Image.open(image_file)
    else:
        img = dataset[i_image][0]  # 1949
    col1, col2 = st.columns(2)
    attn = getPrediction(img, model_name, i_head)
    col1.subheader("Original")
    col1.image(tform(img), use_column_width=True)

    col2.subheader(f"Head Attention {i_head}")
    col2.image(tform(attn), use_column_width=True)


plotAttention()

st.subheader("Prueba en video")
video_file = open("DINO/dino-scratch/app/presidente.mp4", "rb")
video_bytes = video_file.read()

st.video(video_bytes)


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
