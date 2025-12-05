import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
from scipy import stats
from io import BytesIO

from itertools import count

_PLOT_COUNTER = count(1)

def plot(fig, key=None):
    """
    Wrapper para evitar StreamlitDuplicateElementId en apps con muchos Plotly charts.
    Si no se pasa key, genera uno automático único.
    """
    if key is None:
        key = f"plot_{next(_PLOT_COUNTER)}"
    st.plotly_chart(fig, use_container_width=True, key=key)

try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    STATS_MODELS_OK = True
except Exception:
    STATS_MODELS_OK = False


# -----------------------------------------------------
# Configuración
# -----------------------------------------------------
st.set_page_config(page_title="Evaluar para Avanzar - Niza", layout="wide")


# -----------------------------------------------------
# Constantes y utilidades
# -----------------------------------------------------
GRADO_MAP = {
    "Tercero": 3,
    "Cuarto": 4,
    "Quinto": 5,
    "Sexto": 6,
    "Séptimo": 7,
    "Septimo": 7,
    "Octavo": 8,
    "Noveno": 9,
    "Decimo": 10,
    "Décimo": 10,
    "Undecimo": 11,
    "Once": 11
}
GRADO_ORDER = ["Tercero", "Cuarto", "Quinto", "Sexto", "Séptimo", "Octavo", "Noveno", "Decimo", "Undecimo"]


def normalize_grado(s):
    if pd.isna(s):
        return s
    s = str(s).strip()
    s = s.replace("Septimo", "Séptimo")
    s = s.replace("Décimo", "Decimo")
    return s


def make_prueba(name: str) -> str:
    """Deriva 'Prueba' desde QuizName quitando número final y símbolo °."""
    if pd.isna(name):
        return name
    s = str(name).strip()
    s = re.sub(r"\s+\d+\s*°\s*$", "", s)
    s = re.sub(r"\s+\d+\s*$", "", s)
    return s.strip()


def semaforo_accuracy(acc):
    if pd.isna(acc):
        return "Sin dato"
    if acc < 0.55:
        return "Rojo"
    elif acc < 0.65:
        return "Amarillo"
    else:
        return "Verde"


def cohen_d(x, y):
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()
    if len(x) < 2 or len(y) < 2:
        return np.nan
    nx, ny = len(x), len(y)
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    denom = nx + ny - 2
    if denom <= 0:
        return np.nan
    pooled = np.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / denom)
    if pooled == 0 or np.isnan(pooled):
        return np.nan
    return (x.mean() - y.mean()) / pooled


# -----------------------------------------------------
# Carga de datos
# -----------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0)

    expected = [
        "OrgDefinedId", "Genero", "Grado",
        "Antigüedad Innova",
        "IsCorrect", "QuestionId",
        "Competencia",
        "Antigüedad Mentor"
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas esperadas: {missing}")

    df = df.copy()

    df["Grado"] = df["Grado"].apply(normalize_grado)
    df["grado_num"] = df["Grado"].map(GRADO_MAP)

    # Prueba: usar columna del archivo si existe; si no, derivar
    if "Prueba" not in df.columns:
        if "QuizName" not in df.columns:
            raise ValueError("El archivo no trae 'Prueba' ni 'QuizName'.")
        df["Prueba"] = df["QuizName"].apply(make_prueba)

    df["IsCorrect"] = pd.to_numeric(df["IsCorrect"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    df["Antigüedad Innova"] = pd.to_numeric(df["Antigüedad Innova"], errors="coerce")
    df["Antigüedad Mentor"] = pd.to_numeric(df["Antigüedad Mentor"], errors="coerce")
    df["Competencia"] = df["Competencia"].fillna("Sin dato")

    # Exclusión explícita
    if "EdadEst" in df.columns:
        df = df.drop(columns=["EdadEst"])

    return df


def make_student_agg(df_items: pd.DataFrame) -> pd.DataFrame:
    g = df_items.groupby("OrgDefinedId", as_index=False).agg(
        items=("IsCorrect", "size"),
        correct=("IsCorrect", "sum"),
        genero=("Genero", "first"),
        grado=("Grado", "first"),
        grado_num=("grado_num", "first"),
        antig_est=("Antigüedad Innova", "first"),
        antig_mentor=("Antigüedad Mentor", "first")
    )
    g["accuracy"] = g["correct"] / g["items"]
    return g


def kpis(df_items, df_students):
    return {
        "Estudiantes": int(df_students["OrgDefinedId"].nunique()),
        "Ítems": int(len(df_items)),
        "Pruebas": int(df_items["Prueba"].nunique()),
        "Competencias": int(df_items["Competencia"].nunique()),
        "Accuracy ítem": float(df_items["IsCorrect"].mean()),
        "Accuracy estudiante (media)": float(df_students["accuracy"].mean()),
        "Accuracy estudiante (mediana)": float(df_students["accuracy"].median())
    }


# -----------------------------------------------------
# Defaults / reset
# -----------------------------------------------------
def set_defaults(df):
    grados = [g for g in GRADO_ORDER if g in df["Grado"].dropna().unique().tolist()]
    pruebas = sorted(df["Prueba"].dropna().unique().tolist())
    generos = sorted(df["Genero"].dropna().unique().tolist())
    comps = sorted(df["Competencia"].dropna().unique().tolist())
    antig_est_vals = sorted(df["Antigüedad Innova"].dropna().unique().tolist())
    antig_mentor_vals = sorted(df["Antigüedad Mentor"].dropna().unique().tolist())

    st.session_state["grados_sel"] = grados
    st.session_state["pruebas_sel"] = pruebas
    st.session_state["generos_sel"] = generos
    st.session_state["comp_sel"] = comps
    st.session_state["antig_est_sel"] = antig_est_vals
    st.session_state["antig_mentor_sel"] = antig_mentor_vals


# -----------------------------------------------------
# Exportación
# -----------------------------------------------------
def build_export_excel(tables: dict) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, table in tables.items():
            safe_name = sheet_name[:31]
            table.to_excel(writer, index=False, sheet_name=safe_name)
    return output.getvalue()


# -----------------------------------------------------
# UI principal
# -----------------------------------------------------
st.title("Evaluar para Avanzar de Niza - Tablero Institucional")
st.caption(
    "Filtros simultáneos por **Grado** y **Prueba**. "
    "El filtro de **Competencia** solo muestra competencias realmente asociadas "
    "a la selección de grados y pruebas. Sin EdadEst ni Curso. Solo agregados."
)

excel_path = "DatAvanzar.xlsx"
df = load_data(excel_path)

if "initialized" not in st.session_state:
    set_defaults(df)
    st.session_state["initialized"] = True


# -----------------------------------------------------
# Sidebar
# -----------------------------------------------------
with st.sidebar:
    st.header("Filtros")

    grados_opts = [g for g in GRADO_ORDER if g in df["Grado"].dropna().unique().tolist()]
    pruebas_opts = sorted(df["Prueba"].dropna().unique().tolist())
    generos_opts = sorted(df["Genero"].dropna().unique().tolist())
    antig_est_opts = sorted(df["Antigüedad Innova"].dropna().unique().tolist())
    antig_mentor_opts = sorted(df["Antigüedad Mentor"].dropna().unique().tolist())

    st.multiselect("Grado", options=grados_opts, key="grados_sel")
    st.multiselect("Prueba", options=pruebas_opts, key="pruebas_sel")
    st.multiselect("Género", options=generos_opts, key="generos_sel")
    st.multiselect("Antigüedad estudiante (años)", options=antig_est_opts, key="antig_est_sel")
    st.multiselect("Antigüedad mentor (años)", options=antig_mentor_opts, key="antig_mentor_sel")

    # ---- Scope base por Grado + Prueba seleccionados ----
    df_scope = df.copy()

    gs = st.session_state.get("grados_sel", [])
    ps = st.session_state.get("pruebas_sel", [])
    ge = st.session_state.get("generos_sel", [])
    ae = st.session_state.get("antig_est_sel", [])
    am = st.session_state.get("antig_mentor_sel", [])

    if gs:
        df_scope = df_scope[df_scope["Grado"].isin(gs)]
    if ps:
        df_scope = df_scope[df_scope["Prueba"].isin(ps)]
    if ge:
        df_scope = df_scope[df_scope["Genero"].isin(ge)]
    if ae:
        df_scope = df_scope[df_scope["Antigüedad Innova"].isin(ae)]
    if am:
        df_scope = df_scope[df_scope["Antigüedad Mentor"].isin(am)]

    comps_options = sorted(df_scope["Competencia"].dropna().unique().tolist())

    # Mantener consistencia del estado frente a opciones dinámicas
    if "comp_sel" not in st.session_state:
        st.session_state["comp_sel"] = comps_options
    else:
        invalid = set(st.session_state["comp_sel"]) - set(comps_options)
        if invalid:
            st.session_state["comp_sel"] = comps_options

    st.multiselect(
        "Competencia (solo asociadas a Grado + Prueba seleccionados)",
        options=comps_options,
        key="comp_sel"
    )

    st.divider()
    st.subheader("Opciones institucionales")
    show_inference = st.checkbox("Inferenciales básicos", value=True)
    show_models = st.checkbox("Modelo logit simple (ítem)", value=False)
    show_alerts = st.checkbox("Alertas semáforo", value=True)

    st.markdown("**Focos de intervención**")
    gap_threshold = st.slider("Umbral brecha absoluta por género", 0.00, 0.20, 0.05, 0.01)
    min_items_alert = st.number_input("Mínimo ítems para evidencia adecuada", 10, 500, 50, 10)

    st.divider()
    if st.button("Restablecer filtros"):
        set_defaults(df)
        st.rerun()


# -----------------------------------------------------
# Aplicar filtros generales
# -----------------------------------------------------
df_f = df.copy()

gs = st.session_state.get("grados_sel", [])
ps = st.session_state.get("pruebas_sel", [])
ge = st.session_state.get("generos_sel", [])
ae = st.session_state.get("antig_est_sel", [])
am = st.session_state.get("antig_mentor_sel", [])
cs = st.session_state.get("comp_sel", [])

if gs:
    df_f = df_f[df_f["Grado"].isin(gs)]
if ps:
    df_f = df_f[df_f["Prueba"].isin(ps)]
if ge:
    df_f = df_f[df_f["Genero"].isin(ge)]
if ae:
    df_f = df_f[df_f["Antigüedad Innova"].isin(ae)]
if am:
    df_f = df_f[df_f["Antigüedad Mentor"].isin(am)]
if cs:
    df_f = df_f[df_f["Competencia"].isin(cs)]

if df_f.empty:
    st.warning("No hay datos con los filtros seleccionados.")
    st.stop()


# -----------------------------------------------------
# Agregados base
# -----------------------------------------------------
students = make_student_agg(df_f)
k = kpis(df_f, students)

# -----------------------------------------------------
# KPIs
# -----------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Estudiantes (únicos)", k["Estudiantes"])
c2.metric("Ítems analizados", k["Ítems"])
c3.metric("Pruebas", k["Pruebas"])
c4.metric("Competencias", k["Competencias"])

c5, c6, c7 = st.columns(3)
c5.metric("Accuracy ítem", f"{k['Accuracy ítem']:.3f}")
c6.metric("Accuracy estudiante (media)", f"{k['Accuracy estudiante (media)']:.3f}")
c7.metric("Accuracy estudiante (mediana)", f"{k['Accuracy estudiante (mediana)']:.3f}")


# -----------------------------------------------------
# Tablas de competencias en 3 niveles
# -----------------------------------------------------
# 1) Grado × Competencia
comp_grado = (
    df_f.groupby(["Grado", "grado_num", "Competencia"], as_index=False)["IsCorrect"]
    .agg(n_items="size", accuracy_item="mean")
    .sort_values("grado_num")
)

# 2) Prueba × Competencia
comp_prueba = (
    df_f.groupby(["Prueba", "Competencia"], as_index=False)["IsCorrect"]
    .agg(n_items="size", accuracy_item="mean")
    .sort_values(["Prueba", "accuracy_item"])
)

# 3) Grado × Prueba × Competencia
comp_grado_prueba = (
    df_f.groupby(["Grado", "grado_num", "Prueba", "Competencia"], as_index=False)["IsCorrect"]
    .agg(n_items="size", accuracy_item="mean")
    .sort_values(["grado_num", "Prueba", "accuracy_item"])
)


# -----------------------------------------------------
# Alertas (3 niveles)
# -----------------------------------------------------
def build_alerts(df_comp, level_cols):
    al = df_comp.copy()
    al["Semaforo"] = al["accuracy_item"].apply(semaforo_accuracy)
    al["Muestra"] = np.where(al["n_items"] < min_items_alert, "Baja", "Adecuada")
    emoji_map = {"Rojo": "🔴", "Amarillo": "🟡", "Verde": "🟢", "Sin dato": "⚪"}
    al["Alerta"] = al["Semaforo"].map(emoji_map)
    return al

alerts_grado = build_alerts(comp_grado, ["Grado", "Competencia"])
alerts_prueba = build_alerts(comp_prueba, ["Prueba", "Competencia"])
alerts_grado_prueba = build_alerts(comp_grado_prueba, ["Grado", "Prueba", "Competencia"])


# -----------------------------------------------------
# Brechas por género para focos (usamos nivel combinado)
# -----------------------------------------------------
seg_comp_gen = (
    df_f.groupby(["Grado", "Prueba", "Competencia", "Genero"], as_index=False)["IsCorrect"]
    .agg(n_items="size", accuracy_item="mean")
)

pivot_gap = seg_comp_gen.pivot_table(
    index=["Grado", "Prueba", "Competencia"],
    columns="Genero",
    values="accuracy_item"
).reset_index()

generos_presentes = sorted(df_f["Genero"].dropna().unique().tolist())
gap_df = pivot_gap.copy()

if len(generos_presentes) >= 2:
    gA, gB = generos_presentes[0], generos_presentes[1]
    if gA in gap_df.columns and gB in gap_df.columns:
        gap_df["gap_genero"] = gap_df[gA] - gap_df[gB]
        gap_df["abs_gap_genero"] = gap_df["gap_genero"].abs()
        gap_df["Genero_A"] = gA
        gap_df["Genero_B"] = gB
    else:
        gap_df["gap_genero"] = np.nan
        gap_df["abs_gap_genero"] = np.nan
        gap_df["Genero_A"] = None
        gap_df["Genero_B"] = None
else:
    gap_df["gap_genero"] = np.nan
    gap_df["abs_gap_genero"] = np.nan
    gap_df["Genero_A"] = None
    gap_df["Genero_B"] = None

alerts_gp_gap = alerts_grado_prueba.merge(
    gap_df[["Grado", "Prueba", "Competencia", "gap_genero", "abs_gap_genero", "Genero_A", "Genero_B"]],
    on=["Grado", "Prueba", "Competencia"],
    how="left"
)


# -----------------------------------------------------
# Exportación Excel
# -----------------------------------------------------
tables_to_export = {
    "Grado_Competencia": comp_grado,
    "Prueba_Competencia": comp_prueba,
    "Grado_Prueba_Competencia": comp_grado_prueba,
    "Alertas_Grado": alerts_grado,
    "Alertas_Prueba": alerts_prueba,
    "Alertas_Grado_Prueba": alerts_gp_gap,
    "Resumen_Estudiantes": (
        students.groupby(["grado", "genero"], as_index=False)["accuracy"]
        .agg(n="count", media="mean", mediana="median", desv="std")
        .sort_values("media", ascending=False)
    )
}

excel_bytes = build_export_excel(tables_to_export)

st.download_button(
    label="📥 Descargar tablas agregadas (Excel)",
    data=excel_bytes,
    file_name="Evaluar_para_Avanzar_Niza_agregados.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


# -----------------------------------------------------
# Tabs
# -----------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Resumen",
    "Grados",
    "Pruebas",
    "Género",
    "Competencias",
    "Alertas",
    "Focos de intervención"
])


# =====================================================
# TAB 1 - Resumen
# =====================================================
with tab1:
    st.subheader("Distribución general del desempeño (nivel estudiante)")
    fig_hist = px.histogram(students, x="accuracy", nbins=30, title="Distribución de accuracy por estudiante")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Desempeño agregado por Prueba (ítem)")
    prueba_item = (
        df_f.groupby("Prueba", as_index=False)["IsCorrect"]
        .agg(n_items="size", accuracy_item="mean")
        .sort_values("accuracy_item", ascending=False)
    )
    st.dataframe(prueba_item, use_container_width=True)

    fig_pr = px.bar(prueba_item, x="Prueba", y="accuracy_item", title="Accuracy por ítem según Prueba")
    fig_pr.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_pr, use_container_width=True)


# =====================================================
# TAB 2 - Grados
# =====================================================
with tab2:
    st.subheader("Desempeño por grado (nivel estudiante)")

    by_grado = (
        students.groupby(["grado", "grado_num"], as_index=False)["accuracy"]
        .agg(n="count", media="mean", mediana="median", desv="std")
        .sort_values("grado_num")
        .rename(columns={"grado": "Grado"})
    )
    st.dataframe(by_grado, use_container_width=True)

    fig = px.line(by_grado, x="Grado", y="media", markers=True, title="Accuracy promedio por Grado")
    plot(fig)

    fig_box = px.box(students.sort_values("grado_num"), x="grado", y="accuracy",
                     title="Distribución de accuracy por Grado")
    fig_box.update_layout(xaxis_title="Grado")
    st.plotly_chart(fig_box, use_container_width=True)

    if show_inference:
        st.subheader("Inferencia: diferencias entre grados (ANOVA)")
        try:
            grados_presentes = by_grado["Grado"].tolist()
            groups = [students.loc[students["grado"] == g, "accuracy"].dropna() for g in grados_presentes]
            valid = [gr for gr in groups if len(gr) >= 5]
            if len(valid) >= 2:
                f_stat, p_val = stats.f_oneway(*valid)
                st.info(f"ANOVA: F = {f_stat:.3f}, p = {p_val:.3e}.")
            else:
                st.warning("Muestras insuficientes por grado para ANOVA.")
        except Exception as e:
            st.warning(f"No fue posible calcular ANOVA: {e}")


# =====================================================
# TAB 3 - Pruebas
# =====================================================
with tab3:
    st.subheader("Desempeño por Prueba (nivel ítem)")

    prueba_item = (
        df_f.groupby("Prueba", as_index=False)["IsCorrect"]
        .agg(n_items="size", accuracy_item="mean")
        .sort_values("accuracy_item", ascending=False)
    )
    st.dataframe(prueba_item, use_container_width=True)

    fig = px.bar(prueba_item, x="Prueba", y="accuracy_item", title="Accuracy por ítem según Prueba")
    fig.update_layout(xaxis_tickangle=-45)
    plot(fig)


# =====================================================
# TAB 4 - Género
# =====================================================
with tab4:
    st.subheader("Desempeño por género (nivel estudiante)")

    by_gen = (
        students.groupby("genero", as_index=False)["accuracy"]
        .agg(n="count", media="mean", mediana="median", desv="std")
        .sort_values("media", ascending=False)
    )
    st.dataframe(by_gen, use_container_width=True)

    fig_gen = px.bar(by_gen, x="genero", y="media", title="Accuracy promedio por género")
    st.plotly_chart(fig_gen, use_container_width=True)

    st.subheader("Género dentro de grado")
    by_grado_gen = (
        students.groupby(["grado", "grado_num", "genero"], as_index=False)["accuracy"]
        .agg(n="count", media="mean")
        .sort_values("grado_num")
    )
    fig = px.bar(by_grado_gen, x="grado", y="media", color="genero", barmode="group",
                 title="Accuracy por Grado y Género (nivel estudiante)")
    fig.update_layout(xaxis_title="Grado")
    plot(fig)

    st.subheader("Género dentro de prueba (nivel ítem)")
    by_prueba_gen = (
        df_f.groupby(["Prueba", "Genero"], as_index=False)["IsCorrect"]
        .agg(n_items="size", accuracy_item="mean")
        .sort_values("accuracy_item", ascending=False)
    )
    fig = px.bar(by_prueba_gen, x="Prueba", y="accuracy_item", color="Genero", barmode="group",
                 title="Accuracy por ítem: Prueba y Género")
    fig.update_layout(xaxis_tickangle=-45)
    plot(fig)

    if show_inference:
        st.subheader("Inferencia: diferencia global por género (Welch)")
        gens = by_gen["genero"].tolist()
        if len(gens) == 2:
            a = students.loc[students["genero"] == gens[0], "accuracy"].dropna()
            b = students.loc[students["genero"] == gens[1], "accuracy"].dropna()
            if len(a) >= 10 and len(b) >= 10:
                t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
                d = cohen_d(a, b)
                st.info(f"T-test: t = {t_stat:.3f}, p = {p_val:.3e}. | Cohen's d ≈ {d:.3f}")
            else:
                st.warning("Muestras insuficientes por género para t-test.")
        else:
            st.caption("El t-test requiere dos categorías de género.")


# =====================================================
# TAB 5 - Competencias
# =====================================================
with tab5:
    st.subheader("Competencias - vistas institucionales")

    vista = st.radio(
        "Ver competencias por:",
        ["Grado × Competencia", "Prueba × Competencia", "Grado × Prueba × Competencia"],
        horizontal=True
    )

    if vista == "Grado × Competencia":
        st.dataframe(comp_grado, use_container_width=True)

        fig = px.bar(
            comp_grado,
            x="Grado",
            y="accuracy_item",
            color="Competencia",
            barmode="group",
            title="Accuracy por ítem: Grado × Competencia"
        )
        plot(fig)

        pivot = comp_grado.pivot(index="Competencia", columns="Grado", values="accuracy_item")
        fig_h = px.imshow(pivot, aspect="auto", title="Heatmap: Competencia × Grado")
        st.plotly_chart(fig_h, use_container_width=True)

    elif vista == "Prueba × Competencia":
        st.dataframe(comp_prueba, use_container_width=True)

        fig = px.bar(
            comp_prueba,
            x="Prueba",
            y="accuracy_item",
            color="Competencia",
            barmode="group",
            title="Accuracy por ítem: Prueba × Competencia"
        )
        fig.update_layout(xaxis_tickangle=-45)
        plot(fig)

        pivot = comp_prueba.pivot(index="Competencia", columns="Prueba", values="accuracy_item")
        fig_h = px.imshow(pivot, aspect="auto", title="Heatmap: Competencia × Prueba")
        st.plotly_chart(fig_h, use_container_width=True)

    else:
        st.dataframe(comp_grado_prueba, use_container_width=True)

        st.caption(
            "Vista combinada útil para ver cómo cambia una competencia dentro de una misma prueba "
            "a través de grados (o viceversa)."
        )


# =====================================================
# TAB 6 - Alertas
# =====================================================
with tab6:
    st.subheader("Alertas institucionales")

    if not show_alerts:
        st.info("Alertas desactivadas.")
    else:
        vista_a = st.radio(
            "Nivel de alerta:",
            ["Grado × Competencia", "Prueba × Competencia", "Grado × Prueba × Competencia"],
            horizontal=True
        )

        if vista_a == "Grado × Competencia":
            st.dataframe(alerts_grado.sort_values(["Semaforo", "accuracy_item"]), use_container_width=True)

            fig = px.scatter(
                alerts_grado,
                x="Grado",
                y="accuracy_item",
                color="Semaforo",
                size="n_items",
                hover_data=["Competencia", "n_items", "Muestra"],
                title="Mapa de alertas: Grado × Competencia"
            )
            plot(fig)

        elif vista_a == "Prueba × Competencia":
            st.dataframe(alerts_prueba.sort_values(["Semaforo", "accuracy_item"]), use_container_width=True)

            fig = px.scatter(
                alerts_prueba,
                x="Prueba",
                y="accuracy_item",
                color="Semaforo",
                size="n_items",
                hover_data=["Competencia", "n_items", "Muestra"],
                title="Mapa de alertas: Prueba × Competencia"
            )
            fig.update_layout(xaxis_tickangle=-45)
            plot(fig)

        else:
            st.dataframe(
                alerts_gp_gap[
                    ["Grado", "Prueba", "Competencia", "n_items", "accuracy_item", "Alerta",
                     "Muestra", "Genero_A", "Genero_B", "gap_genero", "abs_gap_genero"]
                ].sort_values(["Semaforo", "accuracy_item"]),
                use_container_width=True
            )

            fig = px.scatter(
                alerts_gp_gap,
                x="Prueba",
                y="accuracy_item",
                color="Semaforo",
                size="n_items",
                hover_data=["Grado", "Competencia", "Muestra", "abs_gap_genero"],
                title="Mapa de alertas combinado: Grado × Prueba × Competencia"
            )
            fig.update_layout(xaxis_tickangle=-45)
            plot(fig)

        st.caption(
            f"Umbrales sugeridos: 🔴 < 0.55, 🟡 0.55–0.65, 🟢 ≥ 0.65. "
            f"Muestra adecuada: n_items ≥ {min_items_alert}."
        )


# =====================================================
# TAB 7 - Focos de intervención (Top 10)
# =====================================================
with tab7:
    st.subheader("Top 10 focos de intervención")

    st.markdown(
        f"""
Criterios aplicados sobre **Grado × Prueba × Competencia**:
- **Semáforo = Rojo**
- **Muestra = Adecuada** (n_items ≥ {min_items_alert})
- **Brecha absoluta por género ≥ {gap_threshold:.2f}**
- Priorización por **mayor brecha** y **menor accuracy**.
        """
    )

    focos = alerts_gp_gap.copy()
    focos = focos[
        (focos["Semaforo"] == "Rojo") &
        (focos["Muestra"] == "Adecuada") &
        (focos["abs_gap_genero"].fillna(0) >= gap_threshold)
    ].copy()

    focos = focos.sort_values(
        ["abs_gap_genero", "accuracy_item"],
        ascending=[False, True]
    )

    top10 = focos.head(10)

    if top10.empty:
        st.info("No hay focos que cumplan los criterios con los filtros actuales.")
    else:
        st.dataframe(
            top10[["Grado", "Prueba", "Competencia", "n_items", "accuracy_item",
                   "Genero_A", "Genero_B", "gap_genero", "abs_gap_genero", "Alerta"]],
            use_container_width=True
        )

        fig = px.bar(
            top10,
            x="Competencia",
            y="abs_gap_genero",
            color="Prueba",
            title="Top 10 brechas por género en focos rojos (vista combinada)"
        )
        plot(fig)


# -----------------------------------------------------
# Modelo logit simple opcional
# -----------------------------------------------------
if show_models:
    st.divider()
    st.header("Modelo logit simple (ítem) - opcional")

    if not STATS_MODELS_OK:
        st.warning("statsmodels no está disponible. Agrega 'statsmodels' al requirements.txt.")
    else:
        df_glm = df_f.dropna(subset=["grado_num", "Genero", "Antigüedad Innova", "Antigüedad Mentor"]).copy()
        df_glm = df_glm.rename(columns={
            "Genero": "genero",
            "Antigüedad Innova": "antig_est",
            "Antigüedad Mentor": "antig_mentor"
        })

        try:
            glm = smf.glm(
                "IsCorrect ~ grado_num + C(genero) + antig_est + antig_mentor",
                data=df_glm,
                family=sm.families.Binomial()
            ).fit(
                cov_type="cluster",
                cov_kwds={"groups": df_glm["OrgDefinedId"]}
            )

            params = glm.params
            pvals = glm.pvalues

            out = pd.DataFrame({
                "term": params.index,
                "coef_logit": params.values,
                "odds_ratio": np.exp(params.values),
                "p_value": pvals.values
            })

            st.dataframe(out, use_container_width=True)
        except Exception as e:
            st.warning(f"No fue posible estimar el modelo con los filtros actuales: {e}")

