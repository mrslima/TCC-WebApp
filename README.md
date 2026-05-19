# Road Damage Detection Dashboard

Dashboard Streamlit para visualização dos resultados do projeto de detecção de danos em pavimentação usando modelos CNN leves.

## Modelos comparados

- YOLOv8n
- MobileNetV3-SSDLite
- EfficientDet-Lite0
- NanoDet-Plus Lite

## Métricas exibidas

- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall
- F1-score
- FPS
- Latência
- Parâmetros
- FLOPs
- RAM
- Métricas por classe
- Matriz de confusão
- Co-occurrence
- Gráficos comparativos

## Estrutura

```text
.
├── app.py
├── requirements.txt
├── README.md
└── reports/
    ├── tables/
    └── figures/
