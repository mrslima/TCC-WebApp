# Relatório Extra — Análises por Classe, Comparativos e Co-occurrence

## 1. Quantidade de amostras por classe

- Tabela: `/content/road_damage_framework/reports/tables/samples_by_class.csv`

- Figura imagens por classe: `/content/road_damage_framework/reports/figures/samples_images_by_class.png`

- Figura boxes por classe: `/content/road_damage_framework/reports/figures/samples_boxes_by_class.png`


## 2. Métricas por classe por modelo

- Tabela: `/content/road_damage_framework/reports/tables/per_class_metrics_by_model.csv`

- Métricas incluídas: AP@0.5, AP@0.5:0.95, Precision@0.5, Recall@0.5 e F1@0.5.


## 3. Comparativo geral entre modelos

- Tabela: `/content/road_damage_framework/reports/tables/general_model_comparison.csv`


## 4. Co-occurrence

- Ground truth: `/content/road_damage_framework/reports/tables/cooccurrence_ground_truth_test.csv`

- Resumo das co-occurrences por modelo: `/content/road_damage_framework/reports/tables/cooccurrence_summary.csv`


## Observação

As matrizes de co-occurrence dos modelos são calculadas com base nas predições salvas em JSON. A diagonal indica quantas imagens tiveram uma determinada classe predita; os valores fora da diagonal indicam quantas imagens tiveram duas classes preditas simultaneamente.
