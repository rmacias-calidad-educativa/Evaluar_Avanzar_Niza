import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats

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

def eta_squared_from_anova(ss_between, ss_resid):
    denom = ss_between + ss_resid
    if denom == 0:
        return np.nan
    return ss_between / denom

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

    df["IsCorrect"] = pd.to_numeric(df["IsCorrect"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    df["Antigüedad Innova"] = pd.to_numeric(df["Antigüedad Innova"], errors="coerce")
    df["Antigüedad Mentor"] = pd.to_numeric(df["Antigüedad Mentor"], errors="coerce")
    df["Competencia"] = df["Competencia"].fillna("Sin dato")

    # Excluir explícitamente EdadEst y Curso del flujo analítico
    if "EdadEst" in df.columns:
        df = df.drop(columns=["EdadEst"])

    # Curso existe pero NO se usará en análisis institucional
    # NombreMentor tampoco se mostrará en UI.

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
        "Quices": int(df_items["QuizName"].nunique()),
        "Competencias": int(df_items["Competencia"].nunique()),
        "Accuracy ítem": float(df_items["IsCorrect"].mean()),
        "Accuracy estudiante (media)": float(df_students["accuracy"].mean()),
        "Accuracy estudiante (mediana)": float(df_students["accuracy"].median())
    }


# -----------------------------------------------------
# UI principal
# -----------------------------------------------------
st.title("Evaluar para Avanzar de Niza - Tablero Institucional")
st.caption("Análisis agregados por grado, género, competencias, quices y antigüedades. Sin EdadEst ni Curso. Sin PII.")

with st.sidebar:
    st.header("Fuente y filtros")

    excel_path = "DatAvanzar.xlsx"

    try:
        df = load_data(excel_path)
    except Exception as e:
        st.error(f"No se pudo cargar el archivo: {e}")
        st.stop()

    grados = [g for g in GRADO_ORDER if g in df["Grado"].dropna().unique().tolist()]
    grados_sel = st.multiselect("Grado", options=grados, default=grados)

    generos = sorted(df["Genero"].dropna().unique().tolist())
    generos_sel = st.multiselect("Género", options=generos, default=generos)

    quizzes = sorted(df["QuizName"].dropna().unique().tolist())
    quiz_sel = st.multiselect("Quiz", options=quizzes, default=quizzes)

    comps = sorted(df["Competencia"].dropna().unique().tolist())
    comp_sel = st.multiselect("Competencia", options=comps, default=comps)

    antig_est_vals = sorted(df["Antigüedad Innova"].dropna().unique().tolist())
    antig_est_sel = st.multiselect("Antigüedad estudiante (años)", options=antig_est_vals, default=antig_est_vals)

    antig_mentor_vals = sorted(df["Antigüedad Mentor"].dropna().unique().tolist())
    antig_mentor_sel = st.multiselect("Antigüedad mentor (años)", options=antig_mentor_vals, default=antig_mentor_vals)

    st.divider()
    st.subheader("Opciones")

    show_inference = st.checkbox("Mostrar inferenciales", value=True)
    show_effects = st.checkbox("Mostrar brechas estandarizadas", value=True)
    show_alerts = st.checkbox("Mostrar alertas semáforo", value=True)
    show_models = st.checkbox("Mostrar modelos logit", value=True)


# -----------------------------------------------------
# Aplicar filtros
# -----------------------------------------------------
df_f = df.copy()

if grados_sel:
    df_f = df_f[df_f["Grado"].isin(grados_sel)]
if generos_sel:
    df_f = df_f[df_f["Genero"].isin(generos_sel)]
if quiz_sel:
    df_f = df_f[df_f["QuizName"].isin(quiz_sel)]
if comp_sel:
    df_f = df_f[df_f["Competencia"].isin(comp_sel)]
if antig_est_sel:
    df_f = df_f[df_f["Antigüedad Innova"].isin(antig_est_sel)]
if antig_mentor_sel:
    df_f = df_f[df_f["Antigüedad Mentor"].isin(antig_mentor_sel)]

if df_f.empty:
    st.warning("No hay datos con los filtros seleccionados.")
    st.stop()

students = make_student_agg(df_f)
k = kpis(df_f, students)


# -----------------------------------------------------
# KPIs
# -----------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Estudiantes (únicos)", k["Estudiantes"])
c2.metric("Ítems analizados", k["Ítems"])
c3.metric("Quices", k["Quices"])
c4.metric("Competencias", k["Competencias"])

c5, c6, c7 = st.columns(3)
c5.metric("Accuracy ítem", f"{k['Accuracy ítem']:.3f}")
c6.metric("Accuracy estudiante (media)", f"{k['Accuracy estudiante (media)']:.3f}")
c7.metric("Accuracy estudiante (mediana)", f"{k['Accuracy estudiante (mediana)']:.3f}")


# -----------------------------------------------------
# Tabs
# -----------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Resumen general",
    "Grados",
    "Género",
    "Competencias y Quices",
    "Antigüedades",
    "Brechas, modelos y alertas"
])


# =====================================================
# TAB 1
# =====================================================
with tab1:
    st.subheader("Distribución general del desempeño (nivel estudiante)")
    fig_hist = px.histogram(students, x="accuracy", nbins=30, title="Distribución de accuracy por estudiante")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Quiz: desempeño por ítem")
    quiz_item = (
        df_f.groupby("QuizName", as_index=False)["IsCorrect"]
        .agg(n_items="size", accuracy_item="mean")
        .sort_values("accuracy_item", ascending=False)
    )

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Top 10 quices**")
        st.dataframe(quiz_item.head(10), use_container_width=True)
    with colB:
        st.markdown("**Bottom 10 quices**")
        st.dataframe(quiz_item.tail(10), use_container_width=True)

    fig_quiz = px.bar(quiz_item, x="QuizName", y="accuracy_item", title="Accuracy por ítem según Quiz")
    fig_quiz.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_quiz, use_container_width=True)


# =====================================================
# TAB 2 - Grados
# =====================================================
with tab2:
    st.subheader("Desempeño por grado (nivel estudiante)")

    by_grado = (
        students.groupby(["grado", "grado_num"], as_index=False)["accuracy"]
        .agg(n="count", media="mean", mediana="median", desv="std")
        .sort_values("grado_num")
    )

    st.dataframe(by_grado[["grado", "n", "media", "mediana", "desv"]], use_container_width=True)

    fig_g = px.line(by_grado, x="grado", y="media", markers=True, title="Accuracy promedio por grado")
    st.plotly_chart(fig_g, use_container_width=True)

    fig_box = px.box(students.sort_values("grado_num"), x="grado", y="accuracy", title="Distribución de accuracy por grado")
    st.plotly_chart(fig_box, use_container_width=True)

    if show_inference:
        st.subheader("Inferencia: diferencias entre grados (ANOVA)")
        try:
            groups = [
                students.loc[students["grado"] == g, "accuracy"].dropna()
                for g in by_grado["grado"].tolist()
            ]
            valid_groups = [gr for gr in groups if len(gr) >= 5]
            if len(valid_groups) >= 2:
                f_stat, p_val = stats.f_oneway(*valid_groups)
                st.info(f"ANOVA: F = {f_stat:.3f}, p = {p_val:.3e}.")
            else:
                st.warning("Muestras insuficientes por grado para ANOVA con estos filtros.")
        except Exception as e:
            st.warning(f"No fue posible calcular ANOVA: {e}")


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

    st.subheader("Género dentro de cada grado")

    by_grado_gen = (
        students.groupby(["grado", "grado_num", "genero"], as_index=False)["accuracy"]
        .agg(n="count", media="mean")
        .sort_values("grado_num")
    )

    fig_gg = px.bar(by_grado_gen, x="grado", y="media", color="genero", barmode="group",
                    title="Accuracy por grado y género")
    st.plotly_chart(fig_gg, use_container_width=True)

    if show_inference:
        st.subheader("Inferencia: diferencia por género (Welch)")
        gens = by_gen["genero"].tolist()
        if len(gens) == 2:
            a = students.loc[students["genero"] == gens[0], "accuracy"].dropna()
            b = students.loc[students["genero"] == gens[1], "accuracy"].dropna()
            if len(a) >= 10 and len(b) >= 10:
                t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
                st.info(f"T-test: t = {t_stat:.3f}, p = {p_val:.3e}.")
            else:
                st.warning("Muestras insuficientes por género para t-test.")
        else:
            st.caption("El t-test requiere dos categorías de género.")


# =====================================================
# TAB 4 - Competencias y Quices
# =====================================================
with tab4:
    st.subheader("Desempeño por competencia (nivel ítem)")

    comp_item = (
        df_f.groupby("Competencia", as_index=False)["IsCorrect"]
        .agg(n_items="size", accuracy_item="mean")
        .sort_values("accuracy_item", ascending=False)
    )

    st.dataframe(comp_item, use_container_width=True)

    fig_comp = px.bar(comp_item, x="Competencia", y="accuracy_item", title="Accuracy por ítem según competencia")
    fig_comp.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_comp, use_container_width=True)

    st.subheader("Competencia por grado (ítem)")

    comp_grado = (
        df_f.groupby(["Grado", "grado_num", "Competencia"], as_index=False)["IsCorrect"]
        .agg(n_items="size", accuracy_item="mean")
        .sort_values("grado_num")
    )

    fig_cg = px.line(comp_grado, x="Grado", y="accuracy_item", color="Competencia",
                     title="Trayectorias por competencia a través de grados")
    st.plotly_chart(fig_cg, use_container_width=True)

    st.subheader("Heatmap: Competencia × Quiz (ítem)")

    qc = (
        df_f.groupby(["QuizName", "Competencia"], as_index=False)["IsCorrect"]
        .mean().rename(columns={"IsCorrect": "accuracy_item"})
    )
    qc_p = qc.pivot(index="Competencia", columns="QuizName", values="accuracy_item")

    fig_heat = px.imshow(qc_p, aspect="auto", title="Heatmap de accuracy por ítem")
    st.plotly_chart(fig_heat, use_container_width=True)


# =====================================================
# TAB 5 - Antigüedades
# =====================================================
with tab5:
    st.subheader("Antigüedad del estudiante en Innova (nivel estudiante)")

    by_ant_est = (
        students.groupby("antig_est", as_index=False)["accuracy"]
        .agg(n="count", media="mean", mediana="median", desv="std")
        .sort_values("antig_est")
    )
    st.dataframe(by_ant_est, use_container_width=True)

    fig_ae = px.line(by_ant_est, x="antig_est", y="media", markers=True,
                     title="Accuracy promedio por antigüedad del estudiante")
    st.plotly_chart(fig_ae, use_container_width=True)

    st.subheader("Antigüedad del mentor (sin nombres)")

    by_ant_m = (
        students.groupby("antig_mentor", as_index=False)["accuracy"]
        .agg(n="count", media="mean", mediana="median", desv="std")
        .sort_values("antig_mentor")
    )
    st.dataframe(by_ant_m, use_container_width=True)

    fig_am = px.line(by_ant_m, x="antig_mentor", y="media", markers=True,
                     title="Accuracy promedio por antigüedad del mentor")
    st.plotly_chart(fig_am, use_container_width=True)

    st.caption("Interpretar patrones de mentoría controlando siempre por grado/competencia/quiz para evitar sesgos de asignación.")


# =====================================================
# TAB 6 - Brechas, modelos y alertas
# =====================================================
with tab6:
    st.subheader("Brechas estandarizadas (efectos)")

    if show_effects:
        # Cohen's d global por género
        gen_list = sorted(students["genero"].dropna().unique().tolist())
        d_global = np.nan
        d_table = []

        if len(gen_list) == 2:
            g1, g2 = gen_list[0], gen_list[1]
            a = students.loc[students["genero"] == g1, "accuracy"]
            b = students.loc[students["genero"] == g2, "accuracy"]
            d_global = cohen_d(a, b)

        st.markdown("**Cohen’s d global por género (nivel estudiante)**")
        if np.isnan(d_global):
            st.info("No se pudo calcular Cohen’s d global con los filtros actuales.")
        else:
            st.success(f"d ≈ {d_global:.3f} (magnitud esperada: pequeño ~0.2, mediano ~0.5, grande ~0.8).")

        # Cohen's d por género dentro de cada grado
        st.markdown("**Cohen’s d por grado**")
        for g in sorted(students.dropna(subset=["grado_num"])["grado"].unique(), key=lambda x: GRADO_MAP.get(x, 999)):
            sub = students[students["grado"] == g]
            gens = sorted(sub["genero"].dropna().unique().tolist())
            if len(gens) == 2:
                d = cohen_d(sub.loc[sub["genero"] == gens[0], "accuracy"],
                            sub.loc[sub["genero"] == gens[1], "accuracy"])
                d_table.append({"Grado": g, "Género A": gens[0], "Género B": gens[1], "Cohen_d": d, "n": len(sub)})
            else:
                d_table.append({"Grado": g, "Género A": None, "Género B": None, "Cohen_d": np.nan, "n": len(sub)})

        d_df = pd.DataFrame(d_table)
        st.dataframe(d_df, use_container_width=True)

        fig_d = px.bar(
            d_df.dropna(subset=["Cohen_d"]),
            x="Grado",
            y="Cohen_d",
            title="Brecha estandarizada por género dentro de cada grado"
        )
        st.plotly_chart(fig_d, use_container_width=True)

        # ETA² de ANOVA por grado
        if show_inference:
            st.markdown("**Tamaño de efecto por grado (η² del ANOVA)**")
            try:
                # ANOVA con scipy no da SS directamente; aproximamos con modelo lineal si statsmodels está
                if STATS_MODELS_OK:
                    an = smf.ols("accuracy ~ C(grado)", data=students).fit()
                    aov = sm.stats.anova_lm(an, typ=2)
                    ss_between = float(aov.loc["C(grado)", "sum_sq"])
                    ss_resid = float(aov.loc["Residual", "sum_sq"])
                    eta2 = eta_squared_from_anova(ss_between, ss_resid)
                    st.success(f"η² ≈ {eta2:.3f} (proporción de varianza explicada por el grado).")
                else:
                    st.info("Para η² automático, agrega 'statsmodels' al entorno.")
            except Exception as e:
                st.warning(f"No fue posible calcular η²: {e}")
    else:
        st.info("Brechas estandarizadas desactivadas en el sidebar.")


    st.divider()
    st.subheader("Modelos logit (nivel ítem)")

    if show_models:
        if not STATS_MODELS_OK:
            st.warning("statsmodels no está disponible. Agrega 'statsmodels' al requirements.txt.")
        else:
            st.markdown(
                """
Modelos para probabilidad de acierto **sin EdadEst ni Curso**:

- **Modelo 1 (simple):**  
  `IsCorrect ~ grado_num + género + antig_est + antig_mentor`

- **Modelo 2 (controlado):**  
  añade efectos fijos de `Competencia` y `QuizName`.

Ambos con errores robustos **cluster por estudiante**.
                """
            )

            df_glm = df_f.dropna(subset=["grado_num", "Genero", "Antigüedad Innova", "Antigüedad Mentor"]).copy()
            df_glm = df_glm.rename(columns={
                "Genero": "genero",
                "Antigüedad Innova": "antig_est",
                "Antigüedad Mentor": "antig_mentor"
            })

            # --- Modelo 1
            try:
                glm1 = smf.glm(
                    "IsCorrect ~ grado_num + C(genero) + antig_est + antig_mentor",
                    data=df_glm,
                    family=sm.families.Binomial()
                ).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": df_glm["OrgDefinedId"]}
                )

                params = glm1.params
                bse = glm1.bse
                pvals = glm1.pvalues

                out1 = pd.DataFrame({
                    "term": params.index,
                    "coef_logit": params.values,
                    "odds_ratio": np.exp(params.values),
                    "p_value": pvals.values
                })

                st.markdown("**Modelo 1: Logit simple**")
                st.dataframe(out1, use_container_width=True)
            except Exception as e:
                st.warning(f"No fue posible estimar Modelo 1: {e}")

            # --- Modelo 2
            try:
                glm2 = smf.glm(
                    "IsCorrect ~ grado_num + C(genero) + antig_est + antig_mentor + C(Competencia) + C(QuizName)",
                    data=df_glm,
                    family=sm.families.Binomial()
                ).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": df_glm["OrgDefinedId"]}
                )

                # Resumen limpio (solo términos institucionales)
                keep = ["Intercept", "grado_num", "C(genero)[T.Masculino]", "antig_est", "antig_mentor"]
                keep_existing = [k for k in keep if k in glm2.params.index]

                params2 = glm2.params[keep_existing]
                pvals2 = glm2.pvalues[keep_existing]

                out2 = pd.DataFrame({
                    "term": params2.index,
                    "coef_logit": params2.values,
                    "odds_ratio": np.exp(params2.values),
                    "p_value": pvals2.values
                })

                st.markdown("**Modelo 2: Logit con controles de Competencia y Quiz**")
                st.dataframe(out2, use_container_width=True)
                st.caption("Este resumen oculta dummies para mantener lectura institucional.")
            except Exception as e:
                st.warning(f"No fue posible estimar Modelo 2: {e}")
    else:
        st.info("Modelos desactivados en el sidebar.")


    st.divider()
    st.subheader("Alertas institucionales (semáforo) por Grado × Competencia")

    if show_alerts:
        gc = (
            df_f.groupby(["Grado", "grado_num", "Competencia"], as_index=False)["IsCorrect"]
            .agg(n_items="size", accuracy_item="mean")
            .sort_values(["grado_num", "accuracy_item"])
        )

        gc["Semaforo"] = gc["accuracy_item"].apply(semaforo_accuracy)

        # Regla de muestra baja
        gc["Muestra"] = np.where(gc["n_items"] < 50, "Baja", "Adecuada")

        # Emoji para lectura rápida
        emoji_map = {"Rojo": "🔴", "Amarillo": "🟡", "Verde": "🟢", "Sin dato": "⚪"}
        gc["Alerta"] = gc["Semaforo"].map(emoji_map)

        # Tabla institucional
        show_cols = ["Grado", "Competencia", "n_items", "accuracy_item", "Alerta", "Muestra"]
        st.dataframe(gc[show_cols], use_container_width=True)

        # Gráfico: semáforo vs accuracy
        fig_gc = px.scatter(
            gc,
            x="Grado",
            y="accuracy_item",
            color="Semaforo",
            size="n_items",
            hover_data=["Competencia", "n_items", "Muestra"],
            title="Mapa de alertas por competencia dentro de cada grado"
        )
        st.plotly_chart(fig_gc, use_container_width=True)

        st.caption(
            "Umbrales institucionales sugeridos: Rojo < 0.55, Amarillo 0.55–0.65, Verde ≥ 0.65. "
            "Puedes ajustar estos cortes según metas internas."
        )
    else:
        st.info("Alertas desactivadas en el sidebar.")
