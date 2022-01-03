import requests
from PIL import Image
from io import BytesIO

from config import URL


def getPrediction():
    img = Image.open("chicharon.jpeg").convert("RGB")
    byte_io = BytesIO()
    img.save(byte_io, "png")
    byte_io.seek(0)
    response = requests.post(
        URL,
        files={"image_file": ("1.png", byte_io, "image/png")},
        data={"model_name": "Supervisado", "i_head": "1"},
    )
    img = Image.open(BytesIO(response.content))
    img = img.convert("RGB")
    img.show()


getPrediction()
