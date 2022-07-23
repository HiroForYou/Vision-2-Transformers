# Segmentación 3D Multiórgano con Swin UNETR (BTCV Challenge)

## Introducción

El presente trabajo consiste en la implementación de [Vision Transformers](https://arxiv.org/pdf/2010.11929.pdf) en el campo médico, específicamente en la segmentación 3D de múltiples órgnanos del cuerpo humano a partir de tomografías computarizadas.

## Modelo

En este trabajo se utiliza un modelo Swin UNETR (variante de un [Vision Transformers](https://arxiv.org/pdf/2010.11929.pdf)). La arquitectura de Swin UNETR se muestra a continuación:

<div align="center">
    <img align="center" src="assets/ba15cbbe506c33daae4b19ab3d3bff998e93db32.png" alt="model" width="90%">
</div>

## Dataset

## Pipeline

## Resultados

### Inferencia

A continuación se muestran los resultados de la segmentación en una imagen de prueba. Tiempo de inferencia aproximado de 6 segundos por imagen.

<div 
    align="center" 
    style="display: flex; 
           flex-direction: row; 
           justify-content: space-between; 
           margin-left: 20px;
           margin-right: 20px;">
    <div>
        <img align="center" src="assets/source.gif" alt="source">
        <p>Imagen original</p>
    </div>
    <div>
        <img align="center" src="assets/label.gif" alt="label">
        <p>Etiqueta</p>
    </div>
    <div>
        <img align="center" src="assets/predict.gif" alt="source">
        <p>Predicción</p>
    </div>
</div>

### Visualización 3D

Para la visualización de las tomografías junto a sus etiquetas de segmentación se usó la herramienta [Tensorboard 3D](https://github.com/KitwareMedical/tensorboard-plugin-3d).

https://user-images.githubusercontent.com/40742491/180615042-e089daad-b8d4-42cd-914b-cb71ada1ccc4.mp4
