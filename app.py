# ============================================================
# ROAD DAMAGE DETECTION — STREAMLIT DASHBOARD
# Compatível com GitHub + Streamlit Community Cloud
# ============================================================

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# CONFIGURAÇÃO GERAL
# ============================================================

st.set_page_config(
    page_title="Road Damage Detection Dashboard",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"

CLASS_NAMES = ["D00", "D10", "D20", "D40"]

CLASS_DESCRIPTIONS = {
    "D00": "trinca longitudinal",
    "D10": "trinca transversal",
    "D20": "trinca tipo couro de jacaré",
    "D40": "buraco",
}

METRIC_LABELS = {
    "map50": "mAP@0.5",
    "map50_95": "mAP@0.5:0.95",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "fps": "FPS",
    "latency_mean_ms": "Latência média (ms)",
    "params_m": "Parâmetros (M)",
    "flops_g": "FLOPs (G)",
    "model_size_mb": "Tamanho do modelo (MB)",
    "ram_delta_mb": "Delta RAM (MB)",
    "ram_peak_mb": "Pico RAM (MB)",
}


# ============================================================
# ESTILO
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        font-weight: 800;
    }

    .metric-card {
        padding: 1.1rem;
        border-radius: 1rem;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.035);
        box-shadow: 0 0 20px rgba(0,0,0,0.15);
    }

    .analysis-box {
        padding: 1rem 1.2rem;
        border-radius: 0.8rem;
        border-left: 4px solid #ff4b4b;
        background: rgba(255, 75, 75, 0.08);
        margin-top: 0.6rem;
        margin-bottom: 1rem;
        line-height: 1.55;
    }

    .info-box {
        padding: 1rem 1.2rem;
        border-radius: 0.8rem;
        border-left: 4px solid #4b9eff;
        background: rgba(75, 158, 255, 0.08);
        margin-top: 0.6rem;
        margin-bottom: 1rem;
        line-height: 1.55;
    }

    .success-box {
        padding: 1rem 1.2rem;
        border-radius: 0.8rem;
        border-left: 4px solid #25d366;
        background: rgba(37, 211, 102, 0.08);
        margin-top: 0.6rem;
        margin-bottom: 1rem;
        line-height: 1.55;
    }

    .warning-box {
        padding: 1rem 1.2rem;
        border-radius: 0.8rem;
        border-left: 4px solid #f4c430;
        background: rgba(244, 196, 48, 0.08);
        margin-top: 0.6rem;
        margin-bottom: 1rem;
        line-height: 1.55;
    }

    .small-text {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.72);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

@st.cache_data(show_spinner=False)
def load_csv(filename: str) -> pd.DataFrame:
    path = TABLES_DIR / filename

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_markdown_report(filename: str) -> str:
    path = REPORTS_DIR / filename

    if not path.exists():
        return ""

    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def resolve_report_path(value):
    """
    Resolve caminhos salvos nos CSVs.

    Corrige caminhos absolutos antigos do Colab:
    /content/road_damage_framework/reports/figures/x.png

    Para caminhos locais do repositório:
    reports/figures/x.png
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    value = str(value).strip()

    if value == "" or value.lower() in ["nan", "none"]:
        return None

    # Remove prefixo absoluto do Colab.
    value_clean = value.replace("/content/road_damage_framework/", "")

    candidates = [
        Path(value),
        ROOT / value_clean,
        REPORTS_DIR / value_clean,
        FIGURES_DIR / Path(value_clean).name,
        TABLES_DIR / Path(value_clean).name,
    ]

    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file():
                return candidate
        except Exception:
            continue

    # Fallback por nome do arquivo.
    filename = Path(value_clean).name

    if filename:
        matches = list(ROOT.rglob(filename))
        if matches:
            return matches[0]

    return None


def render_analysis(text: str, kind: str = "analysis"):
    css_class = {
        "analysis": "analysis-box",
        "info": "info-box",
        "success": "success-box",
        "warning": "warning-box",
    }.get(kind, "analysis-box")

    st.markdown(
        f"<div class='{css_class}'>{text}</div>",
        unsafe_allow_html=True,
    )


def safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def fmt(value, digits=4):
    value = safe_float(value)

    if np.isnan(value):
        return "—"

    return f"{value:.{digits}f}"


def fmt2(value):
    value = safe_float(value)

    if np.isnan(value):
        return "—"

    return f"{value:.2f}"


def get_metric_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def clean_display_paths(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype(str).str.replace(
                "/content/road_damage_framework/",
                "",
                regex=False,
            )

    return out


def chart_bar(
    df,
    x,
    y,
    title,
    color=None,
    labels=None,
    sort_desc=True,
    height=430,
):
    if df.empty or x not in df.columns or y not in df.columns:
        st.info("Dados insuficientes para gerar este gráfico.")
        return

    plot_df = df.copy()
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
    plot_df = plot_df.dropna(subset=[y])

    if sort_desc:
        plot_df = plot_df.sort_values(y, ascending=False)

    fig = px.bar(
        plot_df,
        x=x,
        y=y,
        color=color,
        title=title,
        labels=labels or {},
        text_auto=".4f",
        height=height,
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis_title=labels.get(x, x) if labels else x,
        yaxis_title=labels.get(y, y) if labels else y,
        legend_title_text=color if color else "",
    )

    st.plotly_chart(fig, use_container_width=True)


def chart_scatter(
    df,
    x,
    y,
    title,
    color="model",
    size=None,
    labels=None,
    height=460,
):
    if df.empty or x not in df.columns or y not in df.columns:
        st.info("Dados insuficientes para gerar este gráfico.")
        return

    plot_df = df.copy()
    plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
    plot_df = plot_df.dropna(subset=[x, y])

    if plot_df.empty:
        st.info("Dados insuficientes para gerar este gráfico.")
        return

    fig = px.scatter(
        plot_df,
        x=x,
        y=y,
        color=color if color in plot_df.columns else None,
        size=size if size in plot_df.columns else None,
        text="model" if "model" in plot_df.columns else None,
        title=title,
        labels=labels or {},
        height=height,
    )

    fig.update_traces(textposition="top center")

    fig.update_layout(
        template="plotly_dark",
        legend_title_text=color if color else "",
    )

    st.plotly_chart(fig, use_container_width=True)


def load_matrix_csv(path_value):
    path = resolve_report_path(path_value)

    if path is None:
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)

        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "class"})
            df = df.set_index("class")

        return df

    except Exception:
        return pd.DataFrame()


def plot_matrix_heatmap(df, title):
    if df.empty:
        st.warning("Matriz não encontrada.")
        return

    z = df.values.astype(float)
    x = list(df.columns)
    y = list(df.index)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale="Reds",
            text=np.round(z, 3),
            texttemplate="%{text}",
            hovertemplate="Real=%{y}<br>Predito=%{x}<br>Valor=%{z}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=520,
        xaxis_title="Predito",
        yaxis_title="Real",
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================

results_df = load_csv("results_all.csv")
if results_df.empty:
    results_df = load_csv("general_model_comparison.csv")

general_df = load_csv("general_model_comparison.csv")
best_df = load_csv("best_by_model.csv")
per_class_df = load_csv("per_class_metrics_by_model.csv")
samples_df = load_csv("samples_by_class.csv")
env_df = load_csv("colab_environment_summary.csv")
resource_df = load_csv("colab_resource_usage_log.csv")
confusion_summary_df = load_csv("confusion_matrix_summary.csv")
cooc_summary_df = load_csv("cooccurrence_summary.csv")
exports_df = load_csv("exports_summary.csv")
sahi_df = load_csv("sahi_yolov8n_results.csv")
artifact_df = load_csv("artifact_check_summary.csv")

if general_df.empty and not results_df.empty:
    general_df = results_df.copy()

if best_df.empty and not general_df.empty:
    metric = get_metric_col(general_df, ["map50", "map50_95"])
    if metric and "model" in general_df.columns:
        best_df = (
            general_df.sort_values(metric, ascending=False)
            .groupby("model", as_index=False)
            .head(1)
        )


# ============================================================
# ANÁLISE TEXTUAL AUTOMÁTICA
# ============================================================

def build_general_analysis(df: pd.DataFrame) -> str:
    if df.empty:
        return "Os arquivos de resultados gerais não foram encontrados. Verifique se `reports/tables/general_model_comparison.csv` ou `results_all.csv` está no repositório."

    lines = []

    if "map50" in df.columns:
        best_map = df.sort_values("map50", ascending=False).iloc[0]
        lines.append(
            f"<b>mAP@0.5:</b> o melhor desempenho geral foi do modelo <b>{best_map['model']}</b>, "
            f"com mAP@0.5 = <b>{fmt(best_map['map50'])}</b>. "
            "Essa métrica indica a qualidade da detecção considerando IoU mínimo de 0.5."
        )

    if "map50_95" in df.columns:
        best_map95 = df.sort_values("map50_95", ascending=False).iloc[0]
        lines.append(
            f"<b>mAP@0.5:0.95:</b> o modelo mais consistente no critério mais rigoroso foi "
            f"<b>{best_map95['model']}</b>, com valor <b>{fmt(best_map95['map50_95'])}</b>. "
            "Essa métrica penaliza mais fortemente caixas mal alinhadas."
        )

    if "fps" in df.columns:
        best_fps = df.sort_values("fps", ascending=False).iloc[0]
        lines.append(
            f"<b>FPS:</b> o modelo mais rápido foi <b>{best_fps['model']}</b>, "
            f"com <b>{fmt2(best_fps['fps'])}</b> FPS. "
            "Essa métrica mede a velocidade de inferência."
        )

    if "latency_mean_ms" in df.columns:
        best_latency = df.sort_values("latency_mean_ms", ascending=True).iloc[0]
        lines.append(
            f"<b>Latência:</b> a menor latência média foi obtida por <b>{best_latency['model']}</b>, "
            f"com <b>{fmt2(best_latency['latency_mean_ms'])} ms</b> por imagem. "
            "Menor latência é melhor para aplicações embarcadas ou quase em tempo real."
        )

    if "model_size_mb" in df.columns:
        smallest = df.sort_values("model_size_mb", ascending=True).iloc[0]
        lines.append(
            f"<b>Tamanho do modelo:</b> o menor modelo em disco foi <b>{smallest['model']}</b>, "
            f"com <b>{fmt2(smallest['model_size_mb'])} MB</b>. "
            "Modelos menores são mais fáceis de portar para dispositivos com pouca memória."
        )

    if "params_m" in df.columns:
        fewest_params = df.sort_values("params_m", ascending=True).iloc[0]
        lines.append(
            f"<b>Parâmetros:</b> o modelo com menor número de parâmetros foi "
            f"<b>{fewest_params['model']}</b>, com <b>{fmt2(fewest_params['params_m'])}M</b>. "
            "Isso indica menor complexidade estrutural."
        )

    if "flops_g" in df.columns:
        lowest_flops = df.sort_values("flops_g", ascending=True).iloc[0]
        lines.append(
            f"<b>FLOPs:</b> o menor custo computacional estimado foi de "
            f"<b>{lowest_flops['model']}</b>, com <b>{fmt2(lowest_flops['flops_g'])} GFLOPs</b>. "
            "Essa métrica ajuda a estimar o esforço de processamento."
        )

    if "map50" in df.columns and "fps" in df.columns:
        best_map = df.sort_values("map50", ascending=False).iloc[0]
        best_fps = df.sort_values("fps", ascending=False).iloc[0]

        if best_map["model"] == best_fps["model"]:
            lines.append(
                f"<b>Trade-off:</b> nesta execução, <b>{best_map['model']}</b> apresentou simultaneamente "
                "o melhor mAP@0.5 e o maior FPS, sendo o candidato mais forte entre os modelos testados."
            )
        else:
            lines.append(
                f"<b>Trade-off:</b> o melhor mAP@0.5 foi de <b>{best_map['model']}</b>, enquanto "
                f"o maior FPS foi de <b>{best_fps['model']}</b>. A escolha final depende do peso dado "
                "à acurácia ou à velocidade."
            )

    return "<br><br>".join(lines)


def build_dataset_analysis(df: pd.DataFrame) -> str:
    if df.empty:
        return "A tabela de amostras por classe não foi encontrada."

    lines = []

    if {"split", "class", "num_boxes"}.issubset(df.columns):
        total_boxes = int(df["num_boxes"].sum())
        total_images = int(df["num_images"].sum()) if "num_images" in df.columns else None

        lines.append(
            f"<b>Distribuição do dataset:</b> o subconjunto processado contém "
            f"<b>{total_boxes}</b> bounding boxes anotadas"
            + (f" e <b>{total_images}</b> ocorrências de imagem por classe." if total_images else ".")
        )

        by_class = df.groupby("class", as_index=False)["num_boxes"].sum()
        major = by_class.sort_values("num_boxes", ascending=False).iloc[0]
        minor = by_class.sort_values("num_boxes", ascending=True).iloc[0]

        lines.append(
            f"<b>Classe mais frequente:</b> <b>{major['class']}</b> "
            f"({CLASS_DESCRIPTIONS.get(major['class'], '')}) com <b>{int(major['num_boxes'])}</b> boxes."
        )

        lines.append(
            f"<b>Classe menos frequente:</b> <b>{minor['class']}</b> "
            f"({CLASS_DESCRIPTIONS.get(minor['class'], '')}) com <b>{int(minor['num_boxes'])}</b> boxes."
        )

        if "split" in df.columns:
            split_summary = df.groupby("split")["num_boxes"].sum().to_dict()
            split_text = ", ".join([f"{k}: {int(v)} boxes" for k, v in split_summary.items()])
            lines.append(
                f"<b>Divisão dos dados:</b> a distribuição por split ficou em {split_text}."
            )

    return "<br><br>".join(lines)


def build_per_class_analysis(df: pd.DataFrame) -> str:
    if df.empty:
        return "As métricas por classe não foram encontradas."

    lines = []

    if {"model", "class", "ap50"}.issubset(df.columns):
        best_rows = df.sort_values("ap50", ascending=False).groupby("class", as_index=False).head(1)

        for _, row in best_rows.iterrows():
            cls = row["class"]
            lines.append(
                f"<b>{cls}</b> ({CLASS_DESCRIPTIONS.get(cls, '')}): melhor AP@0.5 em "
                f"<b>{row['model']}</b>, com <b>{fmt(row['ap50'])}</b>."
            )

        avg_by_class = df.groupby("class", as_index=False)["ap50"].mean()
        hardest = avg_by_class.sort_values("ap50", ascending=True).iloc[0]
        easiest = avg_by_class.sort_values("ap50", ascending=False).iloc[0]

        lines.append(
            f"<b>Classe mais difícil em média:</b> <b>{hardest['class']}</b>, "
            f"com AP@0.5 médio de <b>{fmt(hardest['ap50'])}</b>."
        )

        lines.append(
            f"<b>Classe com melhor resposta média:</b> <b>{easiest['class']}</b>, "
            f"com AP@0.5 médio de <b>{fmt(easiest['ap50'])}</b>."
        )

    return "<br><br>".join(lines)


def build_confusion_analysis(summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        return "O resumo das matrizes de confusão não foi encontrado."

    lines = []

    diag_rows = []

    for _, row in summary_df.iterrows():
        norm_df = load_matrix_csv(row.get("confusion_normalized_csv"))

        if norm_df.empty:
            continue

        class_cols = [c for c in CLASS_NAMES if c in norm_df.columns]
        class_rows = [c for c in CLASS_NAMES if c in norm_df.index]

        diag_values = []

        for cls in class_rows:
            if cls in class_cols:
                diag_values.append(safe_float(norm_df.loc[cls, cls], np.nan))

        bg_values = []
        if "background" in norm_df.columns:
            for cls in class_rows:
                bg_values.append(safe_float(norm_df.loc[cls, "background"], np.nan))

        diag_mean = np.nanmean(diag_values) if diag_values else np.nan
        bg_mean = np.nanmean(bg_values) if bg_values else np.nan

        diag_rows.append({
            "model": row.get("model"),
            "imgsz": row.get("imgsz"),
            "diag_mean": diag_mean,
            "background_mean": bg_mean,
        })

    if diag_rows:
        diag_df = pd.DataFrame(diag_rows)
        best_diag = diag_df.sort_values("diag_mean", ascending=False).iloc[0]
        lowest_bg = diag_df.sort_values("background_mean", ascending=True).iloc[0]

        lines.append(
            f"<b>Diagonal média:</b> o modelo com maior média na diagonal da matriz normalizada foi "
            f"<b>{best_diag['model']}</b>, com <b>{fmt(best_diag['diag_mean'])}</b>. "
            "Valores maiores na diagonal indicam maior proporção de acertos por classe."
        )

        lines.append(
            f"<b>Background:</b> o menor envio médio de objetos reais para background ocorreu em "
            f"<b>{lowest_bg['model']}</b>, com <b>{fmt(lowest_bg['background_mean'])}</b>. "
            "Valores altos na coluna background indicam falsos negativos."
        )

        lines.append(
            "<b>Interpretação:</b> quando a matriz apresenta muitos valores na coluna background, "
            "o modelo está deixando de detectar objetos anotados. Quando há valores fora da diagonal "
            "entre classes, o erro principal é confusão entre tipos de dano."
        )

    else:
        lines.append(
            "As matrizes foram listadas, mas não foi possível calcular a análise automática. "
            "Verifique se os CSVs normalizados estão em `reports/tables/`."
        )

    return "<br><br>".join(lines)


def build_sahi_analysis(sahi: pd.DataFrame, general: pd.DataFrame) -> str:
    if sahi.empty:
        return "A avaliação SAHI não foi encontrada. Quando disponível, ela compara tiling com resize direto."

    lines = []

    row = sahi.iloc[0]
    lines.append(
        f"<b>SAHI/Tiling:</b> foi avaliado o modelo <b>{row.get('model', 'modelo')}</b> "
        f"com slice_size=<b>{row.get('slice_size', '—')}</b> e overlap=<b>{row.get('overlap', '—')}</b>."
    )

    if "map50" in sahi.columns:
        lines.append(
            f"O resultado com SAHI foi mAP@0.5 = <b>{fmt(row.get('map50'))}</b> e "
            f"mAP@0.5:0.95 = <b>{fmt(row.get('map50_95'))}</b>."
        )

    if not general.empty and "model" in general.columns and "map50" in general.columns:
        model = row.get("model", "yolov8n")
        imgsz = row.get("imgsz", None)

        ref = general[general["model"].astype(str) == str(model)]

        if imgsz is not None and "imgsz" in general.columns:
            ref = ref[ref["imgsz"].astype(str) == str(imgsz)]

        if not ref.empty:
            ref_row = ref.iloc[0]
            diff = safe_float(row.get("map50")) - safe_float(ref_row.get("map50"))

            lines.append(
                f"Comparado ao resize direto do mesmo modelo, a diferença em mAP@0.5 foi "
                f"<b>{diff:+.4f}</b>. Valor positivo indica ganho com tiling; valor negativo indica perda."
            )

    lines.append(
        "<b>Interpretação:</b> SAHI tende a ajudar objetos pequenos porque preserva detalhes locais, "
        "mas aumenta o custo de inferência por dividir a imagem em múltiplos recortes."
    )

    return "<br><br>".join(lines)


def build_exports_analysis(df: pd.DataFrame) -> str:
    if df.empty:
        return "A tabela de exportações não foi encontrada."

    lines = []

    if "status" in df.columns:
        ok_count = int((df["status"].astype(str).str.lower() == "ok").sum())
        error_count = int((df["status"].astype(str).str.lower() == "error").sum())

        lines.append(
            f"<b>Exportações:</b> foram registradas <b>{ok_count}</b> exportações com status OK "
            f"e <b>{error_count}</b> com erro."
        )

    if {"model", "format", "status"}.issubset(df.columns):
        ok_df = df[df["status"].astype(str).str.lower() == "ok"].copy()

        if not ok_df.empty:
            formats = sorted(ok_df["format"].astype(str).unique())
            lines.append(
                f"<b>Formatos exportados com sucesso:</b> {', '.join(formats)}."
            )

            if "size_mb" in ok_df.columns:
                ok_size = ok_df.dropna(subset=["size_mb"])

                if not ok_size.empty:
                    smallest = ok_size.sort_values("size_mb", ascending=True).iloc[0]
                    lines.append(
                        f"<b>Menor artefato exportado:</b> <b>{smallest['model']}</b> em formato "
                        f"<b>{smallest['format']}</b>, com <b>{fmt2(smallest['size_mb'])} MB</b>."
                    )

        yolo_exports = df[df["model"].astype(str) == "yolov8n"]

        if not yolo_exports.empty:
            yolo_ok = yolo_exports[yolo_exports["status"].astype(str).str.lower() == "ok"]
            yolo_formats = sorted(yolo_ok["format"].astype(str).unique())

            if yolo_formats:
                lines.append(
                    f"<b>YOLOv8n:</b> apresentou o conjunto de exportações mais completo nesta execução: "
                    f"{', '.join(yolo_formats)}."
                )

    lines.append(
        "<b>Interpretação:</b> exportações ONNX, TFLite e NCNN são importantes para deploy. "
        "Para Raspberry Pi, NCNN e TFLite INT8 são os caminhos mais relevantes; para análise local, "
        "ONNX e TorchScript ajudam na portabilidade."
    )

    return "<br><br>".join(lines)


def build_environment_analysis(env: pd.DataFrame, resource: pd.DataFrame) -> str:
    lines = []

    if not env.empty:
        row = env.iloc[0]

        lines.append(
            f"<b>Ambiente:</b> execução registrada em plataforma <b>{row.get('platform', '—')}</b>, "
            f"com Python <b>{str(row.get('python_version', '—')).split()[0]}</b>."
        )

        if "ram_total_gb" in env.columns:
            lines.append(
                f"<b>RAM total:</b> <b>{fmt2(row.get('ram_total_gb'))} GB</b>. "
                f"No início do registro, o uso estava em <b>{fmt2(row.get('ram_percent'))}%</b>."
            )

        if "gpu_name" in env.columns:
            lines.append(
                f"<b>GPU:</b> <b>{row.get('gpu_name', '—')}</b>, com "
                f"<b>{fmt2(row.get('gpu_total_memory_gb'))} GB</b> de memória."
            )

    if not resource.empty:
        if "ram_used_gb" in resource.columns:
            max_ram = resource["ram_used_gb"].max()
            lines.append(
                f"<b>Uso de RAM durante o pipeline:</b> pico registrado de "
                f"<b>{fmt2(max_ram)} GB</b> de RAM usada no Colab."
            )

        if "gpu_allocated_gb" in resource.columns:
            max_gpu = resource["gpu_allocated_gb"].max()
            lines.append(
                f"<b>Uso de GPU:</b> maior alocação registrada de "
                f"<b>{fmt2(max_gpu)} GB</b>."
            )

    if not lines:
        return "As informações de ambiente e uso de recursos não foram encontradas."

    return "<br><br>".join(lines)


# ============================================================
# HEADER
# ============================================================

st.title("🛣️ Road Damage Detection — Dashboard Comparativo")
st.caption("YOLOv8n · MobileNetV3-SSDLite · EfficientDet-Lite0 · NanoDet-Plus Lite")

if not REPORTS_DIR.exists():
    st.error("A pasta `reports/` não foi encontrada. Verifique se ela foi enviada ao GitHub junto com o app.")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Configuração")

    st.write("**Diretório do app:**")
    st.code(str(ROOT))

    st.write("**Reports:**")
    st.code(str(REPORTS_DIR))

    st.write("**Status dos arquivos:**")
    st.write("Tables:", "✅" if TABLES_DIR.exists() else "❌")
    st.write("Figures:", "✅" if FIGURES_DIR.exists() else "❌")

    if not general_df.empty and "model" in general_df.columns:
        models_available = sorted(general_df["model"].astype(str).unique())
    else:
        models_available = []

    selected_models = st.multiselect(
        "Modelos",
        options=models_available,
        default=models_available,
    )

    if not general_df.empty and "imgsz" in general_df.columns:
        sizes_available = sorted(general_df["imgsz"].dropna().astype(int).unique())
    else:
        sizes_available = []

    selected_sizes = st.multiselect(
        "Resoluções",
        options=sizes_available,
        default=sizes_available,
    )

    st.divider()

    with st.expander("Debug de arquivos"):
        st.write("CSV encontrados:")
        if TABLES_DIR.exists():
            st.write([p.name for p in sorted(TABLES_DIR.glob("*.csv"))])
        else:
            st.write([])

        st.write("Figuras encontradas:")
        if FIGURES_DIR.exists():
            st.write([p.name for p in sorted(FIGURES_DIR.glob("*.png"))])
        else:
            st.write([])


def apply_filters(df):
    if df.empty:
        return df

    out = df.copy()

    if selected_models and "model" in out.columns:
        out = out[out["model"].astype(str).isin(selected_models)]

    if selected_sizes and "imgsz" in out.columns:
        out = out[out["imgsz"].astype(int).isin(selected_sizes)]

    return out


general_filtered = apply_filters(general_df)
results_filtered = apply_filters(results_df)
per_class_filtered = apply_filters(per_class_df)
confusion_filtered = apply_filters(confusion_summary_df)
cooc_filtered = apply_filters(cooc_summary_df)
exports_filtered = apply_filters(exports_df)
sahi_filtered = apply_filters(sahi_df)


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "Visão geral",
        "Ambiente/RAM",
        "Métricas gerais",
        "Métricas por classe",
        "Matriz de confusão",
        "Co-occurrence",
        "SAHI",
        "Exportações",
        "Relatórios",
    ]
)


# ============================================================
# TAB 1 — VISÃO GERAL
# ============================================================

with tabs[0]:
    st.header("Visão geral do experimento")

    if general_filtered.empty:
        st.warning("Não foi possível carregar as métricas gerais.")
    else:
        col1, col2, col3, col4 = st.columns(4)

        if "map50" in general_filtered.columns:
            best_map = general_filtered.sort_values("map50", ascending=False).iloc[0]
            col1.metric(
                "Melhor mAP@0.5",
                fmt(best_map["map50"]),
                best_map["model"],
            )

        if "fps" in general_filtered.columns:
            best_fps = general_filtered.sort_values("fps", ascending=False).iloc[0]
            col2.metric(
                "Maior FPS",
                fmt2(best_fps["fps"]),
                best_fps["model"],
            )

        if "latency_mean_ms" in general_filtered.columns:
            best_latency = general_filtered.sort_values("latency_mean_ms", ascending=True).iloc[0]
            col3.metric(
                "Menor latência",
                f"{fmt2(best_latency['latency_mean_ms'])} ms",
                best_latency["model"],
            )

        if "model_size_mb" in general_filtered.columns:
            smallest = general_filtered.sort_values("model_size_mb", ascending=True).iloc[0]
            col4.metric(
                "Menor modelo",
                f"{fmt2(smallest['model_size_mb'])} MB",
                smallest["model"],
            )

        render_analysis(build_general_analysis(general_filtered), "analysis")

        st.subheader("Tabela geral")
        st.dataframe(clean_display_paths(general_filtered), use_container_width=True)

        st.subheader("Trade-off mAP@0.5 × FPS")
        chart_scatter(
            general_filtered,
            x="fps",
            y="map50",
            title="Trade-off entre qualidade de detecção e velocidade",
            color="model",
            labels={
                "fps": "FPS",
                "map50": "mAP@0.5",
                "model": "Modelo",
            },
        )

        st.subheader("Trade-off mAP@0.5 × Latência")
        chart_scatter(
            general_filtered,
            x="latency_mean_ms",
            y="map50",
            title="Trade-off entre qualidade de detecção e latência",
            color="model",
            labels={
                "latency_mean_ms": "Latência média (ms)",
                "map50": "mAP@0.5",
                "model": "Modelo",
            },
        )

    st.divider()
    st.header("Distribuição do dataset")

    if samples_df.empty:
        st.info("Arquivo `samples_by_class.csv` não encontrado.")
    else:
        render_analysis(build_dataset_analysis(samples_df), "info")

        st.dataframe(samples_df, use_container_width=True)

        if {"split", "class", "num_boxes"}.issubset(samples_df.columns):
            fig = px.bar(
                samples_df,
                x="class",
                y="num_boxes",
                color="split",
                barmode="group",
                title="Quantidade de bounding boxes por classe e split",
                labels={
                    "class": "Classe",
                    "num_boxes": "Número de boxes",
                    "split": "Split",
                },
                height=430,
            )
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        if {"split", "class", "num_images"}.issubset(samples_df.columns):
            fig = px.bar(
                samples_df,
                x="class",
                y="num_images",
                color="split",
                barmode="group",
                title="Quantidade de imagens por classe e split",
                labels={
                    "class": "Classe",
                    "num_images": "Número de imagens",
                    "split": "Split",
                },
                height=430,
            )
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 2 — AMBIENTE/RAM
# ============================================================

with tabs[1]:
    st.header("Ambiente de execução e uso de recursos")

    render_analysis(build_environment_analysis(env_df, resource_df), "info")

    if not env_df.empty:
        st.subheader("Resumo do ambiente")
        st.dataframe(env_df, use_container_width=True)

    if not resource_df.empty:
        st.subheader("Log de uso de recursos")
        st.dataframe(resource_df, use_container_width=True)

        if "stage" in resource_df.columns and "ram_used_gb" in resource_df.columns:
            fig = px.line(
                resource_df,
                x="stage",
                y="ram_used_gb",
                markers=True,
                title="Uso de RAM total ao longo do pipeline",
                labels={
                    "stage": "Etapa",
                    "ram_used_gb": "RAM usada (GB)",
                },
                height=430,
            )
            fig.update_layout(template="plotly_dark", xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)

        if "stage" in resource_df.columns and "process_ram_rss_gb" in resource_df.columns:
            fig = px.line(
                resource_df,
                x="stage",
                y="process_ram_rss_gb",
                markers=True,
                title="RAM usada pelo processo Python",
                labels={
                    "stage": "Etapa",
                    "process_ram_rss_gb": "RAM do processo (GB)",
                },
                height=430,
            )
            fig.update_layout(template="plotly_dark", xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)

        if "stage" in resource_df.columns and "gpu_allocated_gb" in resource_df.columns:
            fig = px.line(
                resource_df,
                x="stage",
                y="gpu_allocated_gb",
                markers=True,
                title="Memória de GPU alocada",
                labels={
                    "stage": "Etapa",
                    "gpu_allocated_gb": "GPU alocada (GB)",
                },
                height=430,
            )
            fig.update_layout(template="plotly_dark", xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Figuras salvas de uso de recursos")

    for filename in [
        "colab_total_ram_usage.png",
        "colab_process_ram_usage.png",
        "colab_gpu_memory_usage.png",
    ]:
        path = FIGURES_DIR / filename
        if path.exists():
            st.image(str(path), caption=filename, use_container_width=True)


# ============================================================
# TAB 3 — MÉTRICAS GERAIS
# ============================================================

with tabs[2]:
    st.header("Comparativo das métricas gerais")

    if general_filtered.empty:
        st.warning("Métricas gerais não encontradas.")
    else:
        render_analysis(build_general_analysis(general_filtered), "analysis")

        st.dataframe(clean_display_paths(general_filtered), use_container_width=True)

        quality_cols = [
            c for c in ["map50", "map50_95", "precision", "recall", "f1"]
            if c in general_filtered.columns
        ]

        if quality_cols:
            st.subheader("Métricas de qualidade")
            melted = general_filtered.melt(
                id_vars=["model"],
                value_vars=quality_cols,
                var_name="metric",
                value_name="value",
            )

            melted["metric"] = melted["metric"].map(
                lambda x: METRIC_LABELS.get(x, x)
            )

            fig = px.bar(
                melted,
                x="model",
                y="value",
                color="metric",
                barmode="group",
                title="Métricas de qualidade por modelo",
                labels={
                    "model": "Modelo",
                    "value": "Valor",
                    "metric": "Métrica",
                },
                height=480,
            )
            fig.update_layout(template="plotly_dark", xaxis_tickangle=-20)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("mAP@0.5 por modelo")
        chart_bar(
            general_filtered,
            x="model",
            y="map50",
            title="mAP@0.5 por modelo",
            labels={
                "model": "Modelo",
                "map50": "mAP@0.5",
            },
        )

        st.subheader("FPS por modelo")
        chart_bar(
            general_filtered,
            x="model",
            y="fps",
            title="FPS por modelo",
            labels={
                "model": "Modelo",
                "fps": "FPS",
            },
        )

        st.subheader("Latência por modelo")
        chart_bar(
            general_filtered,
            x="model",
            y="latency_mean_ms",
            title="Latência média por modelo",
            labels={
                "model": "Modelo",
                "latency_mean_ms": "Latência média (ms)",
            },
            sort_desc=False,
        )

        st.subheader("Tamanho do modelo")
        chart_bar(
            general_filtered,
            x="model",
            y="model_size_mb",
            title="Tamanho do modelo em disco",
            labels={
                "model": "Modelo",
                "model_size_mb": "MB",
            },
            sort_desc=False,
        )


# ============================================================
# TAB 4 — MÉTRICAS POR CLASSE
# ============================================================

with tabs[3]:
    st.header("Métricas por classe")

    if per_class_filtered.empty:
        st.warning("Arquivo `per_class_metrics_by_model.csv` não encontrado.")
    else:
        render_analysis(build_per_class_analysis(per_class_filtered), "analysis")

        st.dataframe(per_class_filtered, use_container_width=True)

        st.subheader("AP@0.5 por classe e modelo")
        fig = px.bar(
            per_class_filtered,
            x="class",
            y="ap50",
            color="model",
            barmode="group",
            title="AP@0.5 por classe",
            labels={
                "class": "Classe",
                "ap50": "AP@0.5",
                "model": "Modelo",
            },
            height=480,
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("F1@0.5 por classe e modelo")
        if "f1_50" in per_class_filtered.columns:
            fig = px.bar(
                per_class_filtered,
                x="class",
                y="f1_50",
                color="model",
                barmode="group",
                title="F1@0.5 por classe",
                labels={
                    "class": "Classe",
                    "f1_50": "F1@0.5",
                    "model": "Modelo",
                },
                height=480,
            )
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recall@0.5 por classe e modelo")
        if "recall50" in per_class_filtered.columns:
            fig = px.bar(
                per_class_filtered,
                x="class",
                y="recall50",
                color="model",
                barmode="group",
                title="Recall@0.5 por classe",
                labels={
                    "class": "Classe",
                    "recall50": "Recall@0.5",
                    "model": "Modelo",
                },
                height=480,
            )
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 5 — MATRIZ DE CONFUSÃO
# ============================================================

with tabs[4]:
    st.header("Matriz de confusão")

    if confusion_filtered.empty:
        st.warning("Arquivo `confusion_matrix_summary.csv` não encontrado.")
    else:
        render_analysis(build_confusion_analysis(confusion_filtered), "analysis")

        st.subheader("Resumo das matrizes")
        st.dataframe(clean_display_paths(confusion_filtered), use_container_width=True)

        for _, row in confusion_filtered.iterrows():
            model = row.get("model", "modelo")
            imgsz = row.get("imgsz", "")

            st.subheader(f"{model} — imgsz={imgsz}")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Matriz de confusão — contagem**")
                cm_df = load_matrix_csv(row.get("confusion_csv"))

                if not cm_df.empty:
                    plot_matrix_heatmap(
                        cm_df,
                        f"Matriz de confusão — {model} — {imgsz}",
                    )
                else:
                    img_path = resolve_report_path(row.get("confusion_png"))

                    if img_path:
                        st.image(str(img_path), use_container_width=True)
                    else:
                        st.warning("Matriz de contagem não encontrada.")
                        st.code(str(row.get("confusion_png")))

            with col2:
                st.markdown("**Matriz de confusão — normalizada**")
                cm_norm_df = load_matrix_csv(row.get("confusion_normalized_csv"))

                if not cm_norm_df.empty:
                    plot_matrix_heatmap(
                        cm_norm_df,
                        f"Matriz normalizada — {model} — {imgsz}",
                    )
                else:
                    img_path = resolve_report_path(row.get("confusion_normalized_png"))

                    if img_path:
                        st.image(str(img_path), use_container_width=True)
                    else:
                        st.warning("Matriz normalizada não encontrada.")
                        st.code(str(row.get("confusion_normalized_png")))


# ============================================================
# TAB 6 — CO-OCCURRENCE
# ============================================================

with tabs[5]:
    st.header("Co-occurrence")

    st.markdown(
        """
        A matriz de co-occurrence indica quais classes aparecem juntas na mesma imagem.
        A diagonal representa a quantidade de imagens em que a classe apareceu.
        Os valores fora da diagonal indicam quantas imagens tiveram duas classes simultaneamente.
        """
    )

    gt_path = TABLES_DIR / "cooccurrence_ground_truth_test.csv"

    if gt_path.exists():
        st.subheader("Ground truth")
        gt_cooc = pd.read_csv(gt_path)

        if "Unnamed: 0" in gt_cooc.columns:
            gt_cooc = gt_cooc.rename(columns={"Unnamed: 0": "class"}).set_index("class")

        plot_matrix_heatmap(gt_cooc, "Co-occurrence — Ground Truth")

    if cooc_filtered.empty:
        st.warning("Arquivo `cooccurrence_summary.csv` não encontrado.")
    else:
        st.subheader("Resumo")
        st.dataframe(clean_display_paths(cooc_filtered), use_container_width=True)

        for _, row in cooc_filtered.iterrows():
            model = row.get("model", "modelo")
            imgsz = row.get("imgsz", "")

            st.subheader(f"{model} — imgsz={imgsz}")

            cooc_csv = resolve_report_path(row.get("cooccurrence_csv"))

            if cooc_csv is not None:
                df = pd.read_csv(cooc_csv)

                if "Unnamed: 0" in df.columns:
                    df = df.rename(columns={"Unnamed: 0": "class"}).set_index("class")

                plot_matrix_heatmap(df, f"Co-occurrence — {model} — {imgsz}")
            else:
                cooc_fig = resolve_report_path(row.get("cooccurrence_fig"))

                if cooc_fig:
                    st.image(str(cooc_fig), use_container_width=True)
                else:
                    st.warning("Co-occurrence não encontrada.")


# ============================================================
# TAB 7 — SAHI
# ============================================================

with tabs[6]:
    st.header("SAHI / Tiling")

    render_analysis(build_sahi_analysis(sahi_filtered, general_filtered), "info")

    if sahi_filtered.empty:
        st.info("Arquivo `sahi_yolov8n_results.csv` não encontrado.")
    else:
        st.dataframe(clean_display_paths(sahi_filtered), use_container_width=True)

        metric_cols = [
            c for c in ["map50", "map50_95", "precision", "recall", "f1"]
            if c in sahi_filtered.columns
        ]

        if metric_cols:
            melted = sahi_filtered.melt(
                id_vars=["model"],
                value_vars=metric_cols,
                var_name="metric",
                value_name="value",
            )

            melted["metric"] = melted["metric"].map(
                lambda x: METRIC_LABELS.get(x, x)
            )

            fig = px.bar(
                melted,
                x="metric",
                y="value",
                color="model",
                title="Métricas do experimento SAHI",
                labels={
                    "metric": "Métrica",
                    "value": "Valor",
                    "model": "Modelo",
                },
                height=430,
            )
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 8 — EXPORTAÇÕES
# ============================================================

with tabs[7]:
    st.header("Exportações e quantização")

    render_analysis(build_exports_analysis(exports_filtered), "analysis")

    if exports_filtered.empty:
        st.warning("Arquivo `exports_summary.csv` não encontrado.")
    else:
        st.dataframe(clean_display_paths(exports_filtered), use_container_width=True)

        if {"format", "status"}.issubset(exports_filtered.columns):
            st.subheader("Status das exportações")

            status_count = (
                exports_filtered
                .groupby(["format", "status"])
                .size()
                .reset_index(name="count")
            )

            fig = px.bar(
                status_count,
                x="format",
                y="count",
                color="status",
                barmode="group",
                title="Status por formato de exportação",
                labels={
                    "format": "Formato",
                    "count": "Quantidade",
                    "status": "Status",
                },
                height=440,
            )
            fig.update_layout(template="plotly_dark", xaxis_tickangle=-25)
            st.plotly_chart(fig, use_container_width=True)

        if "size_mb" in exports_filtered.columns:
            size_df = exports_filtered.dropna(subset=["size_mb"]).copy()

            if not size_df.empty:
                st.subheader("Tamanho dos artefatos exportados")

                fig = px.bar(
                    size_df,
                    x="format",
                    y="size_mb",
                    color="model",
                    barmode="group",
                    title="Tamanho por formato exportado",
                    labels={
                        "format": "Formato",
                        "size_mb": "Tamanho (MB)",
                        "model": "Modelo",
                    },
                    height=460,
                )
                fig.update_layout(template="plotly_dark", xaxis_tickangle=-25)
                st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 9 — RELATÓRIOS
# ============================================================

with tabs[8]:
    st.header("Relatórios gerados")

    st.subheader("Arquivos disponíveis")

    report_files = sorted(REPORTS_DIR.glob("*"))

    file_rows = []

    for p in report_files:
        if p.is_file():
            file_rows.append({
                "arquivo": p.name,
                "tipo": p.suffix,
                "tamanho_kb": round(p.stat().st_size / 1024, 2),
            })

    st.dataframe(pd.DataFrame(file_rows), use_container_width=True)

    st.subheader("Relatório comparativo Markdown")

    md = load_markdown_report("relatorio_comparativo_colab.md")

    if md:
        st.markdown(md)
    else:
        st.info("Arquivo `relatorio_comparativo_colab.md` não encontrado.")

    st.subheader("Análise extra")

    extra = load_markdown_report("extra_analysis_report.md")

    if extra:
        st.markdown(extra)
    else:
        st.info("Arquivo `extra_analysis_report.md` não encontrado.")

    if not artifact_df.empty:
        st.subheader("Resumo dos artefatos")
        st.dataframe(clean_display_paths(artifact_df), use_container_width=True)
