# Descripción general del modelo

![image](./assets/swin_unetr.png)
Este repositorio contiene el código para Swin UNETR [1,2]. Swin UNETR es el estado del arte en Medical Segmentation Decathlon (MSD) y Beyond the Cranial Vault (BTCV). En [1], se diseña una metodología novedosa para el pre-entrenamiento del backbone Swin UNETR de manera _self-supervised_. Brindamos la opción de entrenar Swin UNETR mediante _fine-tunning_ a partir de pesos _self-supervised_ pre-entrenados o desde cero.

# Tutorial

En el siguiente enlace se proporciona un tutorial para la segmentación de múltiples órganos BTCV utilizando el modelo Swin UNETR.
[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_btcv_segmentation_3d.ipynb)

# Instalación de dependencias

Las dependencias se pueden instalar usando:

```bash
pip install -r requirements.txt
```

# Modelos

Proporcionamos los pesos pre-entrenados _self-supervised_ para el _backbone_ Swin UNETR (paper CVPR [1]) en este <a href="https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt">link</a>.

Proporcionamos varios modelos pre-entrenados en el conjunto de datos BTCV a continuación.

<table>
  <tr>
    <th>Nombre</th>
    <th>Dice (overlap=0.7)</th>
    <th>Dice (overlap=0.5)</th>
    <th>Tamaño de características</th>
    <th># params (M)</th>
    <th>Pre-entrenamiento Self-Supervised</th>
    <th>Descargar</th>
  </tr>
<tr>
    <td>Swin UNETR/Base</td>
    <td>82.25</td>
    <td>81.86</td>
    <td>48</td>
    <td>62.1</td>
    <td>Yes</td>
    <td><a href="https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt">modelo</a></td>
</tr>

<tr>
    <td>Swin UNETR/Small</td>
    <td>79.79</td>
    <td>79.34</td>
    <td>24</td>
    <td>15.7</td>
    <td>No</td>
    <td><a href="https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.small_5000ep_f24_lr2e-4_pretrained.pt">modelo</a></td>
</tr>

<tr>
    <td>Swin UNETR/Tiny</td>
    <td>72.05</td>
    <td>70.35</td>
    <td>12</td>
    <td>4.0</td>
    <td>No</td>
    <td><a href="https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.tiny_5000ep_f12_lr2e-4_pretrained.pt">modelo</a></td>
</tr>

</table>

# Preparación de los datos

![image](./assets/BTCV_organs.png)

Los datos de entrenamiento son del [BTCV challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/217752).

- Objetivo: 13 órganos abdominales, incluidos 1. Bazo 2. Riñón derecho 3. Riñón izquierdo 4. Vesícula biliar 5. Esófago 6. Hígado 7. Estómago 8. Aorta 9. VCI 10. Venas porta y esplénica 11. Páncreas 12. Glándula suprarrenal derecha 13. Glándula suprarrenal izquierda.
- Tarea: Segmentación
- Modalidad: TC
- Tamaño: 30 volúmenes 3D (24 Entrenamiento + 6 Pruebas)

Descargue el archivo json que se usa para entrenar nuestros modelos desde este <a href="https://drive.google.com/file/d/1t4fIQQkONv7ArTSZe4Nucwkk1KfdUDvW/view?usp=sharing"> enlace</a>.

Una vez que se haya descargado el archivo json, colóquelo en la misma carpeta que el dataset. Tenga en cuenta que debe proporcionar la ubicación de su directorio de conjunto de datos utilizando `--data_dir`.

# Entrenamiento

Una red Swin UNETR con hiperparámetros estándar para la segmentación semántica de múltiples órganos (dataset BTCV) se define como:

```bash
model = SwinUNETR(img_size=(96,96,96),
                  in_channels=1,
                  out_channels=14,
                  feature_size=48,
                  use_checkpoint=True,
                  )
```

El modelo Swin UNETR anterior se utiliza para imágenes CT (entrada de 1 canal) con un tamaño de imagen de entrada de `(96, 96, 96)`, salidas de segmentación de `14` clases y un tamaño de características de `48`.
Se pueden encontrar más detalles en [1]. Además, `use_checkpoint=True` permite el uso de _gradient checkpointing_ para el entrenamiento eficiente de la memoria.

Usando los valores predeterminados para los hiperparámetros, el siguiente comando se puede usar para iniciar el entrenamiento usando el paquete AMP nativo de PyTorch:

```bash
python main.py
--feature_size=32
--batch_size=1
--logdir=unetr_test
--fold=0
--optim_lr=1e-4
--lrschedule=warmup_cosine
--infer_overlap=0.5
--save_checkpoint
--data_dir=/dataset/dataset0/
```

## Entrenamiento a partir de pesos _self-supervised_ en una sola GPU (modelo base con _gradient checkpointing_)

Para entrenar un `Swin UNETR` con pesos del encoder _self-supervised_ en una sola GPU con _gradient checkpointing_:

```bash
python main.py --json_list=<json-path> --data_dir=<data-path> --feature_size=48 --use_ssl_pretrained --roi_x=96 --roi_y=96 --roi_z=96  --use_checkpoint --batch_size=<batch-size> --max_epochs=<total-num-epochs> --save_checkpoint
```

## Entrenamiento a partir de pesos _self-supervised_ en múltiples GPU (modelo base sin _gradient checkpointing_)

Para entrenar un `Swin UNETR` con pesos del encoder _self-supervised_ en varias GPU sin _gradient checkpointing_

```bash
python main.py --json_list=<json-path> --data_dir=<data-path> --feature_size=48 --use_ssl_pretrained --roi_x=96 --roi_y=96 --roi_z=96  --distributed --optim_lr=2e-4 --batch_size=<batch-size> --max_epochs=<total-num-epochs> --save_checkpoint
```

## Entrenando desde cero en una sola GPU (modelo base sin AMP)

Para entrenar un `Swin UNETR` desde cero en una GPU sin AMP:

```bash
python main.py --json_list=<json-path> --data_dir=<data-path> --feature_size=48 --noamp\
--roi_x=96 --roi_y=96 --roi_z=96  --use_checkpoint --batch_size=<batch-size> --max_epochs=<total-num-epochs> --save_checkpoint
```

## Entrenando desde cero en una sola GPU (modelo small sin check-pointing)

Para entrenar un `Swin UNETR` desde cero en una GPU sin AMP:

```bash
python main.py --json_list=<json-path> --data_dir=<data-path> --feature_size=24\
--roi_x=96 --roi_y=96 --roi_z=96 --batch_size=<batch-size> --max_epochs=<total-num-epochs> --save_checkpoint
```

## Entrenando desde cero en una sola GPU (modelo tiny sin check-pointing)

Para entrenar un `Swin UNETR` desde cero en una GPU sin AMP:

```bash
python main.py --json_list=<json-path> --data_dir=<data-path> --feature_size=12\
--roi_x=96 --roi_y=96 --roi_z=96 --batch_size=<batch-size> --max_epochs=<total-num-epochs> --save_checkpoint
```

# Evaluación

Para evaluar un `Swin UNETR` en una sola GPU, coloque el checkpoint del modelo en la carpeta `pretrained_models` y proporcione su nombre usando `--pretrained_model_name`:

```bash
python test.py --json_list=<json-path> --data_dir=<data-path> --feature_size=<feature-size>\
--infer_overlap=0.5 --pretrained_model_name=<model-name>
```

# Finetuning

Descargue los checkpoints para los modelos presentados en la tabla anterior y coloque los checkpoints en la carpeta `pretrained_models`.
Utilice los siguientes comandos para el _finetuning_.

## Finetuning de modelo base en una sola GPU (_gradient check-pointing_)

Para hacer finetune a un modelo base de `Swin UNETR` en una sola GPU con _gradient check-pointing_:

```bash
python main.py --json_list=<json-path> --data_dir=<data-path> --feature_size=48 \
--pretrained_model_name='swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt' --resume_ckpt --use_checkpoint \
--batch_size=<batch-size> --max_epochs=<total-num-epochs> --save_checkpoint
```

## Finetuning de modelo small en una sola GPU (_gradient check-pointing_)

Para hacer finetune a un modelo small de `Swin UNETR` en una sola GPU con _gradient check-pointing_:

```bash
python main.py --json_list=<json-path> --data_dir=<data-path> --feature_size=24 \
--pretrained_model_name='swin_unetr.small_5000ep_f24_lr2e-4_pretrained.pt' --resume_ckpt --use_checkpoint \
--batch_size=<batch-size> --max_epochs=<total-num-epochs> --save_checkpoint
```

## Finetuning de modelo tiny en una sola GPU (_gradient check-pointing_)

Para hacer finetune a un modelo tiny de `Swin UNETR` en una sola GPU con _gradient check-pointing_:

```bash
python main.py --json_list=<json-path> --data_dir=<data-path> --feature_size=12 \
--pretrained_model_name='swin_unetr.tiny_5000ep_f12_lr2e-4_pretrained.pt' --resume_ckpt --use_checkpoint \
--batch_size=<batch-size> --max_epochs=<total-num-epochs> --save_checkpoint
```

# Salida de la segmentación

Al seguir los comandos para evaluar `Swin UNETR` de arriba, `test.py` guarda los resultados de la segmentación en el espacio original en una nueva carpeta basada en el nombre del experimento que se pasa en `--exp_name`.

# Citación

Si encuentra útil este repositorio, considere citar los siguientes papers:

```
@inproceedings{tang2022self,
  title={Self-supervised pre-training of swin transformers for 3d medical image analysis},
  author={Tang, Yucheng and Yang, Dong and Li, Wenqi and Roth, Holger R and Landman, Bennett and Xu, Daguang and Nath, Vishwesh and Hatamizadeh, Ali},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20730--20740},
  year={2022}
}

@article{hatamizadeh2022swin,
  title={Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images},
  author={Hatamizadeh, Ali and Nath, Vishwesh and Tang, Yucheng and Yang, Dong and Roth, Holger and Xu, Daguang},
  journal={arXiv preprint arXiv:2201.01266},
  year={2022}
}
```

# Referencias

[1]: Tang, Y., Yang, D., Li, W., Roth, H.R., Landman, B., Xu, D., Nath, V. and Hatamizadeh, A., 2022. Self-supervised pre-training of swin transformers for 3d medical image analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 20730-20740).

[2]: Hatamizadeh, A., Nath, V., Tang, Y., Yang, D., Roth, H. and Xu, D., 2022. Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images. arXiv preprint arXiv:2201.01266.
