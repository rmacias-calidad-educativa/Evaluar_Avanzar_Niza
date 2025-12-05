import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
from scipy import stats
from io import BytesIO

try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    STATS_MODELS_OK = True
except Exception:
    STATS_MODELS_OK = False


# -----------------------------------------------------
# Configuración
# -----------------------------------------------------
st.set_page_config(
    page_title="Evaluar para Avanzar - Niza",
    layout="wide"
)


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
    """
    if pd.isna(name):
        return name
    s = str(name).strip()
    s = re.sub(r"\s+\d+\s*°\s*$", "", s)   # ' 10°'
    s = re.sub(r"\s+\d+\s*$", "", s)      # ' 10'
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
    denom = (nx + ny - 2)
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

    # Columnas mínimas necesarias para el tablero
    base_expected = [
        "OrgDefinedId", "Genero", "Grado",
        "Antigüedad Innova",
        "IsCorrect", "QuestionId",
        "Competencia",
        "Antigüedad Mentor"
    ]
    missing = [c for c in base_expected if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas esperadas: {missing}")

    df = df.copy()

    # Normalizaciones
    df["Grado"] = df["Grado"].apply(normalize_grado)
    df["grado_num"] = df["Grado"].map(GRADO_MAP)

    # Prueba: prioriza columna existente en archivo actualizado
    if "Prueba" not in df.columns:
        if "QuizName" not in df.columns:
            raise ValueError("El archivo no trae 'Prueba' ni 'QuizName'.")
        df["Prueba"] = df["QuizName"].apply(make_prueba)

    # Limpieza binaria
    df["IsCorrect"] = pd.to_numeric(df["IsCorrect"], errors="coerce").fillna(0).astype(int).clip(0, 1)

    # Numericidades
    df["Antigüedad Innova"] = pd.to_numeric(df["Antigüedad Innova"], errors="coerce")
    df["Antigüedad Mentor"] = pd.to_numeric(df["Antigüedad Mentor"], errors="coerce")

    df["Competencia"] = df["Competencia"].fillna("Sin dato")

    # Exclusión explícita por instrucción
    if "EdadEst" in df.columns:
        df = df.drop(columns=["EdadEst"])

    # Curso puede existir, pero NO se usa en análisis institucional
    return df


def make_student_agg(df_items: pd.DataFrame) -> pd.DataFrame:
    """
    Agregado interno a nivel estudiante.
    NO se muestra OrgDefinedId en UI.
    """
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

    st.session_state["modo"] = "Grado"
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
    "Segmentación obligatoria por **Grado** o por **Prueba**. "
    "Las competencias visibles y las alertas se restringen a lo asociado a la selección. "
    "Sin EdadEst ni Curso. Solo agregados."
)

excel_path = "DatAvanzar.xlsx"

try:
    df = load_data(excel_path)
except Exception as e:
    st.error(f"No se pudo cargar el archivo: {e}")
    st.stop()

if "initialized" not in st.session_state:
    set_defaults(df)
    st.session_state["initialized"] = True


# -----------------------------------------------------
# Sidebar
# -----------------------------------------------------
with st.sidebar:
    st.header("Segmentación y filtros")

    # Segmentación obligatoria
    modo = st.radio(
        "Segmentación principal (obligatoria)",
        ["Grado", "Prueba"],
        index=0 if st.session_state.get("modo", "Grado") == "Grado" else 1,
        key="modo"
    )

    st.divider()

    # Filtro principal según modo
    if modo == "Grado":
        grados_opts = [g for g in GRADO_ORDER if g in df["Grado"].dropna().unique().tolist()]
        st.multiselect("Grado", options=grados_opts, key="grados_sel")

        df_scope = df[df["Grado"].isin(st.session_state.get("grados_sel", []))] \
            if st.session_state.get("grados_sel") else df.iloc[0:0]
    else:
        pruebas_opts = sorted(df["Prueba"].dropna().unique().tolist())
        st.multiselect("Prueba", options=pruebas_opts, key="pruebas_sel")

        df_scope = df[df["Prueba"].isin(st.session_state.get("pruebas_sel", []))] \
            if st.session_state.get("pruebas_sel") else df.iloc[0:0]

    # Filtros secundarios
    generos_opts = sorted(df["Genero"].dropna().unique().tolist())
    st.multiselect("Género", options=generos_opts, key="generos_sel")

    antig_est_opts = sorted(df["Antigüedad Innova"].dropna().unique().tolist())
    st.multiselect("Antigüedad estudiante (años)", options=antig_est_opts, key="antig_est_sel")

    antig_mentor_opts = sorted(df["Antigüedad Mentor"].dropna().unique().tolist())
    st.multiselect("Antigüedad mentor (años)", options=antig_mentor_opts, key="antig_mentor_sel")

    # --- Competencias dependientes del scope ---
    df_scope2 = df_scope.copy()
    if st.session_state.get("generos_sel"):
        df_scope2 = df_scope2[df_scope2["Genero"].isin(st.session_state["generos_sel"])]
    if st.session_state.get("antig_est_sel"):
        df_scope2 = df_scope2[df_scope2["Antigüedad Innova"].isin(st.session_state["antig_est_sel"])]
    if st.session_state.get("antig_mentor_sel"):
        df_scope2 = df_scope2[df_scope2["Antigüedad Mentor"].isin(st.session_state["antig_mentor_sel"])]

    comps_options = sorted(df_scope2["Competencia"].dropna().unique().tolist())

    # Asegurar consistencia del estado con opciones dinámicas
    if "comp_sel" not in st.session_state:
        st.session_state["comp_sel"] = comps_options
    else:
        invalid = set(st.session_state["comp_sel"]) - set(comps_options)
        if invalid:
            st.session_state["comp_sel"] = comps_options

    st.multiselect(
        "Competencia (solo asociadas a la selección)",
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
# Aplicar filtros
# -----------------------------------------------------
df_f = df.copy()

if modo == "Grado":
    sel = st.session_state.get("grados_sel", [])
    if sel:
        df_f = df_f[df_f["Grado"].isin(sel)]
else:
    sel = st.session_state.get("pruebas_sel", [])
    if sel:
        df_f = df_f[df_f["Prueba"].isin(sel)]

gen_sel = st.session_state.get("generos_sel", [])
if gen_sel:
    df_f = df_f[df_f["Genero"].isin(gen_sel)]

antig_est_sel = st.session_state.get("antig_est_sel", [])
if antig_est_sel:
    df_f = df_f[df_f["Antigüedad Innova"].isin(antig_est_sel)]

antig_mentor_sel = st.session_state.get("antig_mentor_sel", [])
if antig_mentor_sel:
    df_f = df_f[df_f["Antigüedad Mentor"].isin(antig_mentor_sel)]

comp_sel = st.session_state.get("comp_sel", [])
if comp_sel:
    df_f = df_f[df_f["Competencia"].isin(comp_sel)]

if df_f.empty:
    st.warning("No hay datos con los filtros seleccionados.")
    st.stop()

SEG_COL = "Grado" if modo == "Grado" else "Prueba"


# -----------------------------------------------------
# Agregados
# -----------------------------------------------------
students = make_student_agg(df_f)
k = kpis(df_f, students)

# Tabla central: SEG_COL × Competencia
comp_seg = (
    df_f.groupby([SEG_COL, "Competencia"], as_index=False)["IsCorrect"]
    .agg(n_items="size", accuracy_item="mean")
)

# Alertas semáforo
alerts = comp_seg.copy()
alerts["Semaforo"] = alerts["accuracy_item"].apply(semaforo_accuracy)
alerts["Muestra"] = np.where(alerts["n_items"] < min_items_alert, "Baja", "Adecuada")
emoji_map = {"Rojo": "🔴", "Amarillo": "🟡", "Verde": "🟢", "Sin dato": "⚪"}
alerts["Alerta"] = alerts["Semaforo"].map(emoji_map)

# SEG_COL × Competencia × Género (ítem)
seg_comp_gen = (
    df_f.groupby([SEG_COL, "Competencia", "Genero"], as_index=False)["IsCorrect"]
    .agg(n_items="size", accuracy_item="mean")
)

# Brecha simple por género dentro de cada SEG_COL × Competencia
pivot_gap = seg_comp_gen.pivot_table(
    index=[SEG_COL, "Competencia"],
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

alerts_gap = alerts.merge(
    gap_df[[SEG_COL, "Competencia", "gap_genero", "abs_gap_genero", "Genero_A", "Genero_B"]],
    on=[SEG_COL, "Competencia"],
    how="left"
)


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
# Exportación Excel (agregados)
# -----------------------------------------------------
tables_to_export = {
    f"{SEG_COL}_Competencia": comp_seg.sort_values([SEG_COL, "accuracy_item"]),
    "Alertas_Semaforo_Brechas": alerts_gap.sort_values(["Semaforo", "accuracy_item"]),
    "Seg_Comp_Genero_Item": seg_comp_gen.sort_values([SEG_COL, "Competencia"]),
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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Resumen",
    "Segmentación principal",
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
# TAB 4 - Competencias (solo asociadas a SEG_COL)
# =====================================================
with tab4:
    st.subheader(f"Competencias segmentadas por {SEG_COL}")

    st.dataframe(
        comp_seg.sort_values([SEG_COL, "accuracy_item"]),
        use_container_width=True
    )

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

    pivot = comp_seg.pivot(index="Competencia", columns=SEG_COL, values="accuracy_item")
    fig_h = px.imshow(pivot, aspect="auto", title=f"Heatmap: Competencia × {SEG_COL}")
    st.plotly_chart(fig_h, use_container_width=True)


# =====================================================
# TAB 5 - Alertas (solo asociadas a SEG_COL)
# =====================================================
with tab5:
    st.subheader(f"Alertas institucionales por {SEG_COL} × Competencia")

    if not show_alerts:
        st.info("Alertas desactivadas.")
    else:
        st.dataframe(
            alerts_gap[[SEG_COL, "Competencia", "n_items", "accuracy_item",
                        "Alerta", "Muestra", "Genero_A", "Genero_B",
                        "gap_genero", "abs_gap_genero"]]
            .sort_values(["Semaforo", "accuracy_item"]),
            use_container_width=True
        )

        fig = px.scatter(
            alerts_gap,
            x=SEG_COL,
            y="accuracy_item",
            color="Semaforo",
            size="n_items",
            hover_data=["Competencia", "n_items", "Muestra", "abs_gap_genero"],
            title=f"Mapa de alertas: {SEG_COL} × Competencia"
        )
        if SEG_COL == "Prueba":
            fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"Umbrales sugeridos: 🔴 < 0.55, 🟡 0.55–0.65, 🟢 ≥ 0.65. "
            f"Muestra adecuada: n_items ≥ {min_items_alert}."
        )


# =====================================================
# TAB 6 - Focos de intervención (Top 10)
# =====================================================
with tab6:
    st.subheader("Top 10 focos de intervención")

    st.markdown(
        f"""
Criterios:
- **Semáforo = Rojo**
- **Muestra = Adecuada** (n_items ≥ {min_items_alert})
- **Brecha absoluta por género ≥ {gap_threshold:.2f}**
- Ordenado por **mayor brecha** y **menor accuracy**.
        """
    )

    focos = alerts_gap.copy()
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
            top10[[SEG_COL, "Competencia", "n_items", "accuracy_item",
                   "Genero_A", "Genero_B", "gap_genero", "abs_gap_genero", "Alerta"]],
            use_container_width=True
        )

        fig = px.bar(
            top10,
            x="Competencia",
            y="abs_gap_genero",
            color=SEG_COL,
            title=f"Top focos por brecha de género en rojos ({SEG_COL})"
        )
        st.plotly_chart(fig, use_container_width=True)


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
