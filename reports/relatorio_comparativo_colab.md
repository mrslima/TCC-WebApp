# Relatório comparativo — 4 modelos leves no Colab

Esta versão desconsidera Raspberry Pi e executa os experimentos somente no Google Colab.

## Configuração

- Device: `cuda`

- Dataset reduzido: `True`

- Limite de imagens: `5.00 GB`

- Resoluções avaliadas: `[320]`

- Épocas rápidas: `30`


## Resultados consolidados

| model               |   imgsz | status   |      map50 |   map50_95 |   precision |      recall |          f1 |     fps |   latency_mean_ms |   params_m |   flops_g |   model_size_mb |   ram_delta_mb |   ram_peak_mb | weights_path                                                                           | notes                                                                        |
|:--------------------|--------:|:---------|-----------:|-----------:|------------:|------------:|------------:|--------:|------------------:|-----------:|----------:|----------------:|---------------:|--------------:|:---------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------|
| yolov8n             |     320 | ok       | 0.217217   | 0.0897447  |  0.0670349  | 0.0983522   | 0.0797284   | 66.3299 |           15.0762 |    3.00643 |  1.01105  |        5.92015  |  -3112.15      |       6632.96 | /content/road_damage_framework/runs/yolov8n_320/weights/best.pt                        | nan                                                                          |
| mobilenetv3_ssdlite |     320 | ok       | 0.126383   | 0.0441042  |  0.0310811  | 0.0616832   | 0.0413345   | 26.0444 |           38.396  |    3.7589  |  0.533533 |       14.6385   |      0         |       3482.58 | /content/road_damage_framework/runs/mobilenetv3_ssdlite_320/best.pt                    | nan                                                                          |
| efficientdet_lite0  |     320 | ok       | 0.0389295  | 0.0137264  |  0.00917944 | 0.0234051   | 0.013187    | 37.9221 |           26.3699 |    3.02615 |  0.756032 |       11.8135   |      0.0976562 |       3585.29 | /content/road_damage_framework/runs/efficientdet_lite0_320/efficientdet_lite0_colab.pt | Adapter Colab funcional com backbone EfficientNet-Lite0 e cabeça densa leve. |
| nanodet_plus        |     320 | ok       | 0.00320922 | 0.00182636 |  0.00097796 | 0.000712369 | 0.000824299 | 44.1868 |           22.6312 |    0.16988 |  0.318048 |        0.703403 |      0         |       3545.46 | /content/road_damage_framework/runs/nanodet_plus_320/nanodet_plus_lite_colab.pt        | nan                                                                          |


## Observações

- YOLOv8n usa Ultralytics e formato YOLO.

- MobileNetV3-SSDLite usa Torchvision e formato COCO.

- EfficientDet-Lite0 usa adapter Colab funcional com backbone EfficientNet-Lite0 e cabeça densa leve.

- NanoDet-Plus usa versão PyTorch leve inspirada no NanoDet-Plus, adequada para futura portabilidade.

- A seção de Raspberry Pi foi comentada/desconsiderada nesta versão.
