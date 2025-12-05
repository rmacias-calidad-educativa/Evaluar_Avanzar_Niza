import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import re

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
    """
    Deriva 'Prueba' desde QuizName quitando el número final y el símbolo ° si existe.
    Ej:
      'Inglés 10°' -> 'Inglés'
      'Lenguaje 8' -> 'Lenguaje'
      'Sociales y Ciudadanas 10°' -> 'Sociales y Ciudadanas'
    """
    if pd.isna(name):
        return name
    s = str(name).strip()

    # Quitar patrón típico final con grado
    s = re.sub(r"\s+\d+\s*°\s*$", "", s)   # ' 10°'
    s = re.sub(r"\s+\d+\s*$", "", s)       # ' 10'

    return s.strip()

def cohen_d(x, y):
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()
    if len(x) < 2 or len(y) < 2:
        return np.nan
    nx, ny = len(x), len(y)
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    pooled = np.sqrt(((nx-1)*sx**2 + (ny-1)*sy**2) / (nx+ny-2)) if (nx+ny-2) > 0 else np.nan
    if pooled == 0 or np.isnan(pooled):
        return np.nan
    return (x.mean() - y.mean()) / pooled

def semaforo_accuracy(acc):
    if pd.isna(acc):
        return "Sin dato"
    if acc < 0.55:
        return "Rojo"
    elif acc < 0.65:
        return "Amarillo"
    else:
        return "Verde"


# -----------------------------------------------------
# Carga de datos
# -----------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0)

    expected = [
        "OrgDefinedId", "Genero", "Grado",
        "Antigüedad Innova", "QuizName",
        "IsCorrect", "QuestionId",
        "Competencia", "NombreMentor",
        "Antigüedad Mentor"
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas esperadas: {missing}")

    df = df.copy()
    df["Grado"] = df["Grado"].apply(normalize_grado)
    df["grado_num"] = df["Grado"].map(GRADO_MAP)

    # Crear Prueba derivada del QuizName
    df["Prueba"] = df["QuizName"].apply(make_prueba)

    df["IsCorrect"] = pd.to_numeric(df["IsCorrect"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    df["Antigüedad Innova"] = pd.to_numeric(df["Antigüedad Innova"], errors="coerce")
    df["Antigüedad Mentor"] = pd.to_numeric(df["Antigüedad Mentor"], errors="coerce")
    df["Competencia"] = df["Competencia"].fillna("Sin dato")

    # Excluir explícitamente EdadEst
    if "EdadEst" in df.columns:
        df = df.drop(columns=["EdadEst"])

    # Curso se deja pero NO se usa en el tablero por instrucción.
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
# UI principal
# -----------------------------------------------------
st.title("Evaluar para Avanzar de Niza - Tablero Institucional")
st.caption("Segmentación obligatoria por Grado o por Prueba. Competencias y alertas se calculan solo con esa elección. Sin EdadEst ni Curso. Sin PII.")


with st.sidebar:
    st.header("Fuente y segmentación")

    excel_path = "DatAvanzar.xlsx"
    try:
        df = load_data(excel_path)
    except Exception as e:
        st.error(f"No se pudo cargar el archivo: {e}")
        st.stop()

    # --------- Segmentación obligatoria ----------
    modo = st.radio(
        "Segmentación principal (obligatoria)",
        ["Grado", "Prueba"],
        index=0
    )

    st.divider()
    st.subheader("Filtros")

    # Filtros de segmentación (mutuamente excluyentes)
    if modo == "Grado":
        grados = [g for g in GRADO_ORDER if g in df["Grado"].dropna().unique().tolist()]
        grados_sel = st.multiselect("Grado", options=grados, default=grados)
        pruebas_sel = None
    else:
        pruebas = sorted(df["Prueba"].dropna().unique().tolist())
        pruebas_sel = st.multiselect("Prueba", options=pruebas, default=pruebas)
        grados_sel = None

    # Filtros secundarios (sí aplican en ambos modos)
    generos = sorted(df["Genero"].dropna().unique().tolist())
    generos_sel = st.multiselect("Género", options=generos, default=generos)

    comps = sorted(df["Competencia"].dropna().unique().tolist())
    comp_sel = st.multiselect("Competencia", options=comps, default=comps)

    antig_est_vals = sorted(df["Antigüedad Innova"].dropna().unique().tolist())
    antig_est_sel = st.multiselect("Antigüedad estudiante (años)", options=antig_est_vals, default=antig_est_vals)

    antig_mentor_vals = sorted(df["Antigüedad Mentor"].dropna().unique().tolist())
    antig_mentor_sel = st.multiselect("Antigüedad mentor (años)", options=antig_mentor_vals, default=antig_mentor_vals)

    st.divider()
    st.subheader("Opciones")

    show_inference = st.checkbox("Mostrar inferenciales", value=True)
    show_models = st.checkbox("Mostrar logit simple", value=False)
    show_alerts = st.checkbox("Mostrar alertas semáforo", value=True)


# -----------------------------------------------------
# Aplicar filtros
# -----------------------------------------------------
df_f = df.copy()

if modo == "Grado" and grados_sel:
    df_f = df_f[df_f["Grado"].isin(grados_sel)]

if modo == "Prueba" and pruebas_sel:
    df_f = df_f[df_f["Prueba"].isin(pruebas_sel)]

if generos_sel:
    df_f = df_f[df_f["Genero"].isin(generos_sel)]

if comp_sel:
    df_f = df_f[df_f["Competencia"].isin(comp_sel)]

if antig_est_sel:
    df_f = df_f[df_f["Antigüedad Innova"].isin(antig_est_sel)]

if antig_mentor_sel:
    df_f = df_f[df_f["Antigüedad Mentor"].isin(antig_mentor_sel)]

if df_f.empty:
    st.warning("No hay datos con los filtros seleccionados.")
    st.stop()


# -----------------------------------------------------
# Agregados base
# -----------------------------------------------------
students = make_student_agg(df_f)
k = kpis(df_f, students)

# Definir columna de segmentación para todo el tablero
SEG_COL = "Grado" if modo == "Grado" else "Prueba"


# -----------------------------------------------------
# KPIs
# -----------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Estudiantes (únicos)", k["Estudiantes"])
c2.metric("Ítems analizados", k["Ítems"])
c3.metric("Pruebas (agrupadas)", k["Pruebas"])
c4.metric("Competencias", k["Competencias"])

c5, c6, c7 = st.columns(3)
c5.metric("Accuracy ítem", f"{k['Accuracy ítem']:.3f}")
c6.metric("Accuracy estudiante (media)", f"{k['Accuracy estudiante (media)']:.3f}")
c7.metric("Accuracy estudiante (mediana)", f"{k['Accuracy estudiante (mediana)']:.3f}")


# -----------------------------------------------------
# Tabs
# -----------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Resumen",
    "Segmentación principal",
    "Género",
    "Competencias",
    "Alertas"
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
# TAB 2 - Segmentación principal
# =====================================================
with tab2:
    st.subheader(f"Desempeño por {SEG_COL}")

    if modo == "Grado":
        by_seg = (
            students.groupby(["grado", "grado_num"], as_index=False)["accuracy"]
            .agg(n="count", media="mean", mediana="median", desv="std")
            .sort_values("grado_num")
            .rename(columns={"grado": "Grado"})
        )

        st.dataframe(by_seg[["Grado", "n", "media", "mediana", "desv"]], use_container_width=True)

        fig = px.line(by_seg, x="Grado", y="media", markers=True, title="Accuracy promedio por Grado")
        st.plotly_chart(fig, use_container_width=True)

        fig_box = px.box(students.sort_values("grado_num"), x="grado", y="accuracy",
                         title="Distribución de accuracy por Grado")
        fig_box.update_layout(xaxis_title="Grado")
        st.plotly_chart(fig_box, use_container_width=True)

        if show_inference:
            st.subheader("Inferencia: diferencias entre grados (ANOVA)")
            try:
                grados_presentes = by_seg["Grado"].tolist()
                groups = [students.loc[students["grado"] == g, "accuracy"].dropna() for g in grados_presentes]
                valid = [gr for gr in groups if len(gr) >= 5]
                if len(valid) >= 2:
                    f_stat, p_val = stats.f_oneway(*valid)
                    st.info(f"ANOVA: F = {f_stat:.3f}, p = {p_val:.3e}.")
                else:
                    st.warning("Muestras insuficientes por grado para ANOVA.")
            except Exception as e:
                st.warning(f"No fue posible calcular ANOVA: {e}")

    else:
        # Por Prueba usamos nivel ítem (más estable que intentar "accuracy por prueba" por estudiante)
        by_seg = (
            df_f.groupby("Prueba", as_index=False)["IsCorrect"]
            .agg(n_items="size", accuracy_item="mean")
            .sort_values("accuracy_item", ascending=False)
        )

        st.dataframe(by_seg, use_container_width=True)

        fig = px.bar(by_seg, x="Prueba", y="accuracy_item", title="Accuracy por ítem según Prueba")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)


# =====================================================
# TAB 3 - Género
# =====================================================
with tab3:
    st.subheader("Desempeño por género (nivel estudiante)")

    by_gen = (
        students.groupby("genero", as_index=False)["accuracy"]
        .agg(n="count", media="mean", mediana="median", desv="std")
        .sort_values("media", ascending=False)
    )
    st.dataframe(by_gen, use_container_width=True)

    fig_gen = px.bar(by_gen, x="genero", y="media", title="Accuracy promedio por género")
    st.plotly_chart(fig_gen, use_container_width=True)

    st.subheader(f"Género dentro de {SEG_COL}")

    if modo == "Grado":
        by_seg_gen = (
            students.groupby(["grado", "grado_num", "genero"], as_index=False)["accuracy"]
            .agg(n="count", media="mean")
            .sort_values("grado_num")
        )
        fig = px.bar(by_seg_gen, x="grado", y="media", color="genero", barmode="group",
                     title="Accuracy por Grado y Género")
        fig.update_layout(xaxis_title="Grado")
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Por Prueba usamos nivel ítem para el cruce con género
        by_seg_gen = (
            df_f.groupby(["Prueba", "Genero"], as_index=False)["IsCorrect"]
            .agg(n_items="size", accuracy_item="mean")
            .sort_values("accuracy_item", ascending=False)
        )
        fig = px.bar(by_seg_gen, x="Prueba", y="accuracy_item", color="Genero", barmode="group",
                     title="Accuracy por ítem según Prueba y Género")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    if show_inference:
        st.subheader("Inferencia: diferencia por género (Welch)")
        gens = by_gen["genero"].tolist()
        if len(gens) == 2:
            a = students.loc[students["genero"] == gens[0], "accuracy"].dropna()
            b = students.loc[students["genero"] == gens[1], "accuracy"].dropna()
            if len(a) >= 10 and len(b) >= 10:
                t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
                d = cohen_d(a, b)
                st.info(f"T-test: t = {t_stat:.3f}, p = {p_val:.3e}.  |  Cohen's d ≈ {d:.3f}")
            else:
                st.warning("Muestras insuficientes por género para t-test.")
        else:
            st.caption("El t-test requiere dos categorías de género.")


# =====================================================
# TAB 4 - Competencias (solo por Grado o por Prueba)
# =====================================================
with tab4:
    st.subheader(f"Competencias segmentadas por {SEG_COL}")

    # Tabla central: SEG_COL × Competencia
    comp_seg = (
        df_f.groupby([SEG_COL, "Competencia"], as_index=False)["IsCorrect"]
        .agg(n_items="size", accuracy_item="mean")
        .sort_values("accuracy_item")
    )
    st.dataframe(comp_seg, use_container_width=True)

    fig = px.bar(
        comp_seg,
        x=SEG_COL,
        y="accuracy_item",
        color="Competencia",
        barmode="group",
        title=f"Accuracy por ítem: {SEG_COL} × Competencia"
    )
    if SEG_COL == "Prueba":
        fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap SEG_COL × Competencia
    pivot = comp_seg.pivot(index="Competencia", columns=SEG_COL, values="accuracy_item")
    fig_h = px.imshow(pivot, aspect="auto", title=f"Heatmap: Competencia × {SEG_COL}")
    st.plotly_chart(fig_h, use_container_width=True)


# =====================================================
# TAB 5 - Alertas (solo por Grado o por Prueba)
# =====================================================
with tab5:
    st.subheader(f"Alertas institucionales por {SEG_COL} × Competencia")

    if not show_alerts:
        st.info("Alertas desactivadas.")
    else:
        al = (
            df_f.groupby([SEG_COL, "Competencia"], as_index=False)["IsCorrect"]
            .agg(n_items="size", accuracy_item="mean")
        )
        al["Semaforo"] = al["accuracy_item"].apply(semaforo_accuracy)
        al["Muestra"] = np.where(al["n_items"] < 50, "Baja", "Adecuada")

        emoji_map = {"Rojo": "🔴", "Amarillo": "🟡", "Verde": "🟢", "Sin dato": "⚪"}
        al["Alerta"] = al["Semaforo"].map(emoji_map)

        st.dataframe(
            al[[SEG_COL, "Competencia", "n_items", "accuracy_item", "Alerta", "Muestra"]]
            .sort_values(["Semaforo", "accuracy_item"]),
            use_container_width=True
        )

        fig = px.scatter(
            al,
            x=SEG_COL,
            y="accuracy_item",
            color="Semaforo",
            size="n_items",
            hover_data=["Competencia", "n_items", "Muestra"],
            title=f"Mapa de alertas: {SEG_COL} × Competencia"
        )
        if SEG_COL == "Prueba":
            fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Umbrales sugeridos: 🔴 < 0.55, 🟡 0.55–0.65, 🟢 ≥ 0.65. "
            "Etiqueta de muestra baja para evitar decisiones con evidencia frágil."
        )


# -----------------------------------------------------
# Modelo logit simple opcional (sin EdadEst ni Curso)
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
