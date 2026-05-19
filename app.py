
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path('/content/road_damage_framework')
REPORTS_DIR = ROOT / 'reports'
TABLES_DIR = REPORTS_DIR / 'tables'
FIGURES_DIR = REPORTS_DIR / 'figures'

st.set_page_config(page_title='Road Damage Detection — Dashboard', page_icon='🛣️', layout='wide')
st.title('🛣️ Road Damage Detection — Dashboard Comparativo')
st.caption('YOLOv8n · MobileNetV3-SSDLite · EfficientDet-Lite0 · NanoDet-Plus Lite')

def load_csv(name):
    path = TABLES_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def show_df(df, title):
    st.subheader(title)
    if df.empty:
        st.info(f'Tabela não encontrada ou vazia: {title}')
    else:
        st.dataframe(df, use_container_width=True)

def show_image(path, caption=None):
    path = Path(path)
    if path.exists():
        st.image(str(path), caption=caption or path.name, use_container_width=True)

results = load_csv('results_all.csv')
best = load_csv('best_by_model.csv')
per_class = load_csv('per_class_metrics_by_model.csv')
samples = load_csv('samples_by_class.csv')
resource_log = load_csv('colab_resource_usage_log.csv')
env = load_csv('colab_environment_summary.csv')
exports = load_csv('exports_summary.csv')
sahi = load_csv('sahi_yolov8n_results.csv')
cooc_summary = load_csv('cooccurrence_summary.csv')
conf_summary = load_csv('confusion_matrix_summary.csv')

with st.sidebar:
    st.header('Filtros')
    if not results.empty and 'imgsz' in results.columns:
        sizes = sorted(results['imgsz'].dropna().unique().tolist())
        selected_sizes = st.multiselect('Resoluções', sizes, default=sizes)
    else:
        selected_sizes = []
    if not results.empty and 'model' in results.columns:
        models = sorted(results['model'].dropna().unique().tolist())
        selected_models = st.multiselect('Modelos', models, default=models)
    else:
        selected_models = []

filtered = results.copy()
if selected_sizes and 'imgsz' in filtered.columns:
    filtered = filtered[filtered['imgsz'].isin(selected_sizes)]
if selected_models and 'model' in filtered.columns:
    filtered = filtered[filtered['model'].isin(selected_models)]

tab_overview, tab_env, tab_metrics, tab_class, tab_conf, tab_cooc, tab_sahi, tab_exports = st.tabs([
    'Visão geral', 'Ambiente/RAM', 'Métricas gerais', 'Métricas por classe',
    'Matriz de confusão', 'Co-occurrence', 'SAHI', 'Exportações'
])

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Modelos', int(filtered['model'].nunique()) if not filtered.empty and 'model' in filtered.columns else 0)
    c2.metric('Experimentos', len(filtered))
    c3.metric('Melhor mAP@0.5', f"{filtered['map50'].max():.4f}" if not filtered.empty and 'map50' in filtered.columns else 'N/A')
    c4.metric('Melhor FPS', f"{filtered['fps'].max():.2f}" if not filtered.empty and 'fps' in filtered.columns else 'N/A')
    show_df(filtered, 'Resultados filtrados')
    if not filtered.empty and {'model','map50'}.issubset(filtered.columns):
        fig = px.bar(filtered, x='model', y='map50', color='model', facet_col='imgsz' if 'imgsz' in filtered.columns else None, title='mAP@0.5 por modelo e resolução')
        st.plotly_chart(fig, use_container_width=True)
    if not filtered.empty and {'fps','map50','model'}.issubset(filtered.columns):
        fig = px.scatter(filtered, x='fps', y='map50', color='model', size='imgsz' if 'imgsz' in filtered.columns else None, hover_data=filtered.columns, title='Trade-off mAP@0.5 × FPS')
        st.plotly_chart(fig, use_container_width=True)

with tab_env:
    show_df(env, 'Configuração do ambiente')
    show_df(resource_log, 'Uso de recursos do Colab')
    if not resource_log.empty:
        for col, title in [('process_ram_rss_gb','RAM do processo'), ('ram_used_gb','RAM total usada'), ('gpu_allocated_gb','Memória GPU alocada')]:
            if col in resource_log.columns:
                fig = px.line(resource_log, x=resource_log.index, y=col, markers=True, hover_data=resource_log.columns, title=title)
                st.plotly_chart(fig, use_container_width=True)

with tab_metrics:
    show_df(best, 'Melhor resultado por modelo')
    show_df(filtered, 'Todas as métricas gerais')
    quality_cols = [c for c in ['map50','map50_95','precision','recall','f1'] if c in filtered.columns]
    if not filtered.empty and quality_cols:
        melt = filtered.melt(id_vars=[c for c in ['model','imgsz'] if c in filtered.columns], value_vars=quality_cols, var_name='metric', value_name='value')
        fig = px.bar(melt, x='model', y='value', color='metric', barmode='group', facet_col='imgsz' if 'imgsz' in melt.columns else None, title='Comparativo de métricas gerais')
        st.plotly_chart(fig, use_container_width=True)

with tab_class:
    show_df(samples, 'Quantidade de amostras por classe')
    show_df(per_class, 'Métricas por classe por modelo')
    if not per_class.empty and {'model','class','ap50'}.issubset(per_class.columns):
        fig = px.bar(per_class, x='class', y='ap50', color='model', barmode='group', facet_col='imgsz' if 'imgsz' in per_class.columns else None, title='AP@0.5 por classe')
        st.plotly_chart(fig, use_container_width=True)
    if not per_class.empty and {'model','class','f1_50'}.issubset(per_class.columns):
        fig = px.bar(per_class, x='class', y='f1_50', color='model', barmode='group', facet_col='imgsz' if 'imgsz' in per_class.columns else None, title='F1@0.5 por classe')
        st.plotly_chart(fig, use_container_width=True)

with tab_conf:
    show_df(conf_summary, 'Resumo das matrizes de confusão')
    if not conf_summary.empty:
        for _, row in conf_summary.iterrows():
            st.markdown(f"### {row.get('model','')} — imgsz={row.get('imgsz','')}")
            col1, col2 = st.columns(2)
            with col1: show_image(row.get('confusion_png',''), 'Matriz absoluta')
            with col2: show_image(row.get('confusion_normalized_png',''), 'Matriz normalizada')

with tab_cooc:
    show_df(cooc_summary, 'Resumo de co-occurrence')
    show_image(FIGURES_DIR / 'cooccurrence_ground_truth_test.png', 'Co-occurrence Ground Truth')
    if not cooc_summary.empty:
        for _, row in cooc_summary.iterrows():
            show_image(row.get('cooccurrence_fig',''), f"Co-occurrence — {row.get('model','')} — {row.get('imgsz','')}")

with tab_sahi:
    show_df(sahi, 'Resultados SAHI / tiling')
    if not sahi.empty and {'imgsz','map50'}.issubset(sahi.columns):
        fig = px.line(sahi, x='imgsz', y='map50', markers=True, title='SAHI — mAP@0.5 por resolução')
        st.plotly_chart(fig, use_container_width=True)

with tab_exports:
    show_df(exports, 'Exportações e quantizações')
    if not exports.empty and {'format','status'}.issubset(exports.columns):
        fig = px.histogram(exports, x='format', color='status', barmode='group', title='Status das exportações por formato')
        st.plotly_chart(fig, use_container_width=True)
