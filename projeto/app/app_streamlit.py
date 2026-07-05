# -*- coding: utf-8 -*-
"""
App interativo do Trabalho Final de Análise Inferencial (Streamlit).

Permite manipular o tamanho da amostra, a semente, o nível de confiança e o
tipo de teste/distribuição, visualizando tabelas, métricas e gráficos.

Execução (a partir da pasta raiz do trabalho):
    streamlit run projeto/app/app_streamlit.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sps
import streamlit as st
from scipy.stats import chi2, chi2_contingency, norm, t
from statsmodels.stats.power import GofChisquarePower
from statsmodels.stats.proportion import proportion_confint

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Análise Inferencial — RH", layout="wide")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(os.path.dirname(APP_DIR))
DADOS_POP = os.path.join(RAIZ, "RH - Novas opor", "archive", "aug_train.csv")

ROTULO_TARGET = {0.0: "Não busca mudança", 1.0: "Busca mudança"}
GRUPO_EMPRESA = {
    "Pvt Ltd": "Empresa privada",
    "Funded Startup": "Startup",
    "Early Stage Startup": "Startup",
    "Public Sector": "Setor público",
    "NGO": "ONG/Outros",
    "Other": "ONG/Outros",
}
ORDEM_EXPERIENCIA = ["<1", "1-5", "6-10", "11-20", ">20"]

VARS_QUANTITATIVAS = ["training_hours", "city_development_index"]
VARS_CATEGORICAS = [
    "faixa_experiencia", "relevent_experience", "tipo_empresa",
    "sem_info_empresa", "education_level", "enrolled_university",
    "last_new_job", "company_size", "major_discipline", "gender",
]
VARS_BINARIAS_GRUPO = ["target", "relevent_experience", "sem_info_empresa"]


def banda_experiencia(v):
    if pd.isna(v):
        return np.nan
    if v == "<1":
        return "<1"
    if v == ">20":
        return ">20"
    anos = int(v)
    if anos <= 5:
        return "1-5"
    if anos <= 10:
        return "6-10"
    return "11-20"


@st.cache_data
def carregar_populacao():
    pop = pd.read_csv(DADOS_POP)
    pop["faixa_experiencia"] = pop["experience"].map(banda_experiencia)
    pop["tipo_empresa"] = pop["company_type"].map(GRUPO_EMPRESA)
    pop["sem_info_empresa"] = np.where(pop["company_type"].isna(),
                                       "Sem informação", "Com informação")
    pop["grupo_target"] = pop["target"].map(ROTULO_TARGET)
    return pop


def cramer_v(chi2_val, tabela):
    n_tab = tabela.to_numpy().sum()
    k = min(tabela.shape) - 1
    return np.sqrt(chi2_val / (n_tab * k))


def grafico_distribuicao(dist, stat_obs, alpha, alternativa, nome):
    """Densidade da distribuição de referência com região crítica sombreada."""
    if nome == "chi2":
        x_max = max(dist.ppf(0.999), stat_obs * 1.15)
        xs = np.linspace(0.001, x_max, 500)
        crit = dist.ppf(1 - alpha)
        regioes = [(crit, x_max)]
    else:
        lim = max(4.2, abs(stat_obs) + 1)
        xs = np.linspace(-lim, lim, 500)
        if alternativa == "bilateral":
            c = dist.ppf(1 - alpha / 2)
            regioes = [(-lim, -c), (c, lim)]
        elif alternativa == "unilateral à direita":
            regioes = [(dist.ppf(1 - alpha), lim)]
        else:
            regioes = [(-lim, dist.ppf(alpha))]
    ys = dist.pdf(xs)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.plot(xs, ys, color="steelblue")
    for a, b in regioes:
        mask = (xs >= a) & (xs <= b)
        ax.fill_between(xs[mask], ys[mask], color="indianred", alpha=0.45,
                        label="Região crítica")
    ax.axvline(stat_obs, color="black", linestyle="--",
               label=f"Estatística observada = {stat_obs:.3f}")
    manejadores, rotulos = ax.get_legend_handles_labels()
    por_rotulo = dict(zip(rotulos, manejadores))
    ax.legend(por_rotulo.values(), por_rotulo.keys(), fontsize=8)
    ax.set_ylabel("Densidade")
    fig.tight_layout()
    return fig


def mostra_decisao(p_valor, alpha):
    c1, c2 = st.columns(2)
    c1.metric("p-valor", f"{p_valor:.4g}")
    c2.metric("Nível de significância (α)", f"{alpha:.2f}")
    if p_valor < alpha:
        st.error(f"**Decisão: Rejeita H₀** (p-valor = {p_valor:.4g} < α = {alpha:.2f})")
    else:
        st.success(f"**Decisão: Não rejeita H₀** (p-valor = {p_valor:.4g} ≥ α = {alpha:.2f})")


# ---------------------------------------------------------------------------
# Barra lateral — parâmetros globais
# ---------------------------------------------------------------------------
pop = carregar_populacao()
N = len(pop)

st.sidebar.title("⚙️ Parâmetros")
semente = int(st.sidebar.number_input("Semente aleatória", 0, 999_999, 2026))
n = int(st.sidebar.slider("Tamanho da amostra (n)", 50, 5000, 752, step=10))
conf = st.sidebar.select_slider("Nível de confiança",
                                options=[0.90, 0.95, 0.99], value=0.95)
ALPHA = round(1 - conf, 2)
st.sidebar.caption(
    f"População: N = {N:,} | fração amostrada: {n / N:.1%} | α = {ALPHA:.2f}\n\n"
    "A amostra é redesenhada por AAS sem reposição sempre que n ou a semente mudam."
)

amostra = pop.sample(n=n, replace=False, random_state=semente)

# ---------------------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------------------
st.title("Intenção de mudança de emprego — Análise Inferencial")
st.caption(
    "População: HR Analytics — Job Change of Data Scientists (Kaggle), N = 19.158. "
    "Amostragem aleatória simples sem reposição."
)

abas = st.tabs([
    "📌 Visão geral", "📊 Descritiva", "🎯 Estimação (IC)",
    "🧪 Testes de hipóteses", "📐 Planejamento amostral", "✅ Validação",
])

# ---------------------------------------------------------------------------
# Aba 1 — Visão geral
# ---------------------------------------------------------------------------
with abas[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("População (N)", f"{N:,}")
    c2.metric("Amostra (n)", f"{n:,}")
    c3.metric("Buscam mudança (amostra)", f"{amostra['target'].mean():.1%}")
    c4.metric("Média de training_hours", f"{amostra['training_hours'].mean():.1f} h")

    st.subheader("Primeiras linhas da amostra")
    st.dataframe(amostra.head(10), use_container_width=True)

    st.subheader("Dados faltantes na amostra")
    faltas = pd.DataFrame({
        "n ausente": amostra.isna().sum(),
        "% ausente": (amostra.isna().mean() * 100).round(1),
    })
    st.dataframe(faltas[faltas["n ausente"] > 0], use_container_width=True)

    st.download_button(
        "⬇️ Baixar amostra (CSV)",
        amostra.to_csv(index=False).encode("utf-8"),
        file_name=f"amostra_n{n}_semente{semente}.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# Aba 2 — Descritiva
# ---------------------------------------------------------------------------
with abas[1]:
    st.subheader("Variável quantitativa")
    var_q = st.selectbox("Variável", VARS_QUANTITATIVAS, key="desc_quant")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(amostra[var_q].describe().round(3).to_frame("valor"))
        st.metric("Assimetria", f"{amostra[var_q].skew():.2f}")
    with c2:
        bins = st.slider("Número de classes do histograma", 10, 60, 30)
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        ax.hist(amostra[var_q], bins=bins, color="steelblue", edgecolor="white")
        ax.set_xlabel(var_q)
        ax.set_ylabel("Frequência")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.subheader("Variável categórica × intenção de mudança")
    var_c = st.selectbox("Variável", VARS_CATEGORICAS, key="desc_cat")
    if var_c == "gender":
        st.warning("`gender` tem ~24% de dados faltantes e forte desbalanceamento — "
                   "interprete com cautela (ver seção de vieses do relatório).")
    dados_c = amostra.dropna(subset=[var_c])
    freq = pd.DataFrame({
        "Frequência": dados_c[var_c].value_counts(),
        "Proporção": dados_c[var_c].value_counts(normalize=True).round(4),
        "Taxa de busca por mudança":
            dados_c.groupby(var_c)["target"].mean().round(4),
    })
    if var_c == "faixa_experiencia":
        freq = freq.reindex([c for c in ORDEM_EXPERIENCIA if c in freq.index])
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(freq, use_container_width=True)
    with c2:
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        freq["Taxa de busca por mudança"].plot(kind="bar", ax=ax, color="steelblue")
        ax.axhline(amostra["target"].mean(), color="indianred", linestyle="--",
                   label=f"Taxa geral = {amostra['target'].mean():.2f}")
        ax.set_ylabel("Proporção que busca mudança")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ---------------------------------------------------------------------------
# Aba 3 — Estimação intervalar
# ---------------------------------------------------------------------------
with abas[2]:
    alvo = st.radio("Parâmetro a estimar",
                    ["Média (variável quantitativa)",
                     "Proporção que busca mudança"], horizontal=True)
    mostrar_pop = st.checkbox("Mostrar valor populacional verdadeiro (gabarito)", True)

    if alvo.startswith("Média"):
        var_q = st.selectbox("Variável", VARS_QUANTITATIVAS, key="ic_quant")
        xbar = amostra[var_q].mean()
        s = amostra[var_q].std(ddof=1)
        gl = n - 1
        t_crit = t.ppf(1 - ALPHA / 2, df=gl)
        erro = t_crit * s / np.sqrt(n)
        li, ls = xbar - erro, xbar + erro
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimativa pontual (x̄)", f"{xbar:.3f}")
        c2.metric("Margem de erro", f"{erro:.3f}")
        c3.metric(f"IC {conf:.0%} (t-Student, {gl} gl)", f"({li:.3f}; {ls:.3f})")
        st.caption("Fórmula: x̄ ± t(1−α/2; n−1) · s/√n — variância desconhecida.")
        verdadeiro = pop[var_q].mean()
    else:
        metodo = st.selectbox("Método do IC",
                              ["normal (Wald)", "beta (Clopper-Pearson, exato)",
                               "wilson"])
        p_hat = amostra["target"].mean()
        x_suc = int(amostra["target"].sum())
        li, ls = proportion_confint(count=x_suc, nobs=n, alpha=ALPHA,
                                    method=metodo.split(" ")[0])
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimativa pontual (p̂)", f"{p_hat:.4f}")
        c2.metric("Sucessos / n", f"{x_suc} / {n}")
        c3.metric(f"IC {conf:.0%} ({metodo})", f"({li:.4f}; {ls:.4f})")
        st.caption(f"Condição de aproximação normal: n·p̂·(1−p̂) = "
                   f"{n * p_hat * (1 - p_hat):.1f} (≥ 5 é o recomendado).")
        verdadeiro = pop["target"].mean()

    fig, ax = plt.subplots(figsize=(7.5, 1.9))
    ax.errorbar([(li + ls) / 2], [0], xerr=[[(ls - li) / 2]], fmt="o",
                color="steelblue", capsize=6, markersize=8,
                label=f"IC {conf:.0%}")
    if mostrar_pop:
        ax.axvline(verdadeiro, color="indianred", linestyle="--",
                   label=f"Valor populacional = {verdadeiro:.3f}")
        dentro = li <= verdadeiro <= ls
        st.info(f"O intervalo {'**contém** ✅' if dentro else '**não contém** ❌'} "
                "o valor populacional verdadeiro.")
    ax.set_yticks([])
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Aba 4 — Testes de hipóteses
# ---------------------------------------------------------------------------
with abas[3]:
    teste = st.selectbox("Tipo de teste", [
        "t para uma média",
        "z para uma proporção",
        "t de Welch (duas médias independentes)",
        "Qui-quadrado de independência",
    ])
    st.divider()

    if teste == "t para uma média":
        var_q = st.selectbox("Variável", VARS_QUANTITATIVAS, key="th_quant")
        padrao = 60.0 if var_q == "training_hours" else 0.80
        mu0 = st.number_input("Valor de referência (μ₀)", value=padrao,
                              step=1.0 if var_q == "training_hours" else 0.01,
                              format="%.2f")
        alternativa = st.radio("Hipótese alternativa",
                               ["bilateral", "unilateral à direita",
                                "unilateral à esquerda"], horizontal=True)
        xbar = amostra[var_q].mean()
        s = amostra[var_q].std(ddof=1)
        gl = n - 1
        t_obs = (xbar - mu0) / (s / np.sqrt(n))
        dist = t(df=gl)
        if alternativa == "bilateral":
            p_valor = 2 * (1 - dist.cdf(abs(t_obs)))
            simbolo = "≠"
        elif alternativa == "unilateral à direita":
            p_valor = 1 - dist.cdf(t_obs)
            simbolo = ">"
        else:
            p_valor = dist.cdf(t_obs)
            simbolo = "<"
        st.markdown(f"**H₀:** μ = {mu0:g}  |  **H_A:** μ {simbolo} {mu0:g}")
        c1, c2, c3 = st.columns(3)
        c1.metric("x̄", f"{xbar:.3f}")
        c2.metric("s", f"{s:.3f}")
        c3.metric("Estatística t", f"{t_obs:.4f}  (gl = {gl})")
        mostra_decisao(p_valor, ALPHA)
        fig_dist = grafico_distribuicao(dist, t_obs, ALPHA, alternativa, "t")
        st.pyplot(fig_dist)
        plt.close(fig_dist)
        st.caption("Distribuição de referência: t de Student. Com n grande, o TCL "
                   "valida o teste mesmo sem normalidade dos dados.")

    elif teste == "z para uma proporção":
        p0 = st.number_input("Proporção de referência (p₀)", 0.01, 0.99, 0.20,
                             step=0.01, format="%.2f")
        alternativa = st.radio("Hipótese alternativa",
                               ["bilateral", "unilateral à direita",
                                "unilateral à esquerda"], horizontal=True)
        p_hat = amostra["target"].mean()
        z_obs = (p_hat - p0) / np.sqrt(p0 * (1 - p0) / n)
        dist = norm()
        if alternativa == "bilateral":
            p_valor = 2 * (1 - dist.cdf(abs(z_obs)))
            simbolo = "≠"
        elif alternativa == "unilateral à direita":
            p_valor = 1 - dist.cdf(z_obs)
            simbolo = ">"
        else:
            p_valor = dist.cdf(z_obs)
            simbolo = "<"
        st.markdown(f"**H₀:** p = {p0:.2f}  |  **H_A:** p {simbolo} {p0:.2f} "
                    "— proporção que busca mudança (target = 1)")
        ok = (n * p0 >= 5) and (n * (1 - p0) >= 5)
        st.caption(f"Condições: n·p₀ = {n * p0:.0f} e n·(1−p₀) = {n * (1 - p0):.0f} "
                   f"({'ambas ≥ 5 ✅' if ok else 'violadas ❌ — use o teste exato'})")
        c1, c2 = st.columns(2)
        c1.metric("p̂", f"{p_hat:.4f}")
        c2.metric("Estatística z", f"{z_obs:.4f}")
        mostra_decisao(p_valor, ALPHA)
        fig_dist = grafico_distribuicao(dist, z_obs, ALPHA, alternativa, "z")
        st.pyplot(fig_dist)
        plt.close(fig_dist)
        st.caption("Distribuição de referência: Normal padrão (aproximação da binomial).")

    elif teste == "t de Welch (duas médias independentes)":
        var_q = st.selectbox("Variável quantitativa", VARS_QUANTITATIVAS,
                             key="welch_quant")
        var_g = st.selectbox("Variável de agrupamento (binária)",
                             VARS_BINARIAS_GRUPO)
        grupos = sorted(amostra[var_g].dropna().unique(), key=str)
        g1 = (amostra.loc[amostra[var_g] == grupos[0], var_q].dropna()
              if len(grupos) > 0 else pd.Series(dtype=float))
        g2 = (amostra.loc[amostra[var_g] == grupos[1], var_q].dropna()
              if len(grupos) > 1 else pd.Series(dtype=float))
        if len(g1) < 2 or len(g2) < 2:
            st.warning("A amostra atual não possui dois grupos com pelo menos "
                       "duas observações para esta variável de agrupamento — "
                       "aumente o n ou troque a semente na barra lateral.")
        else:
            rot1 = ROTULO_TARGET.get(grupos[0], str(grupos[0]))
            rot2 = ROTULO_TARGET.get(grupos[1], str(grupos[1]))
            alternativa = st.radio("Hipótese alternativa",
                                   ["bilateral", "unilateral à direita",
                                    "unilateral à esquerda"], horizontal=True)
            alt_scipy = {"bilateral": "two-sided",
                         "unilateral à direita": "greater",
                         "unilateral à esquerda": "less"}[alternativa]
            res = sps.ttest_ind(g1, g2, equal_var=False, alternative=alt_scipy)
            st.markdown(f"**H₀:** μ({rot1}) = μ({rot2})  |  "
                        f"grupos definidos por `{var_g}`")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"x̄ — {rot1}", f"{g1.mean():.3f} (n={len(g1)})")
            c2.metric(f"x̄ — {rot2}", f"{g2.mean():.3f} (n={len(g2)})")
            razao = max(g1.std(ddof=1), g2.std(ddof=1)) / min(g1.std(ddof=1),
                                                              g2.std(ddof=1))
            c3.metric("Razão dos desvios-padrão", f"{razao:.2f}")
            c4.metric("Estatística t", f"{res.statistic:.4f} (gl = {res.df:.1f})")
            mostra_decisao(res.pvalue, ALPHA)
            fig_dist = grafico_distribuicao(t(df=res.df), res.statistic, ALPHA,
                                            alternativa, "t")
            st.pyplot(fig_dist)
            plt.close(fig_dist)
            # boxplot com ax.boxplot (robusto a versões do pandas/matplotlib —
            # DataFrame.boxplot(by=<Series>) quebra em versões recentes do pandas)
            fig, ax = plt.subplots(figsize=(7.5, 3.2))
            ax.boxplot([g1.values, g2.values])
            ax.set_xticks([1, 2])
            ax.set_xticklabels([rot1, rot2])
            ax.set_ylabel(var_q)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Distribuição de referência: t de Student com graus de "
                       "liberdade de Welch-Satterthwaite (não assume variâncias "
                       "iguais).")

    else:  # Qui-quadrado de independência
        var_c = st.selectbox("Variável categórica (cruzada com target)",
                             VARS_CATEGORICAS, key="qq_cat")
        if var_c == "gender":
            st.warning("`gender` tem ~24% de dados faltantes e forte "
                       "desbalanceamento — resultado frágil e sensível em "
                       "contexto de RH (ver relatório).")
        dados_qq = amostra.dropna(subset=[var_c])
        tabela = pd.crosstab(dados_qq[var_c], dados_qq["grupo_target"])
        if var_c == "faixa_experiencia":
            # reindexa apenas com as faixas presentes (amostras pequenas podem
            # não conter todas — linhas NaN quebrariam o chi2_contingency)
            tabela = tabela.reindex([c for c in ORDEM_EXPERIENCIA
                                     if c in tabela.index])
        chi2_obs, p_valor, gl_qq, esperadas = chi2_contingency(tabela)
        esp_df = pd.DataFrame(esperadas, index=tabela.index, columns=tabela.columns)
        st.markdown(f"**H₀:** `{var_c}` e a intenção de mudança são independentes  |  "
                    f"**H_A:** existe associação")
        st.caption(f"Casos completos usados: {len(dados_qq)} de {n} "
                   f"({len(dados_qq) / n:.1%} da amostra)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Frequências observadas**")
            st.dataframe(tabela, use_container_width=True)
        with c2:
            st.markdown("**Frequências esperadas sob H₀**")
            st.dataframe(esp_df.round(1), use_container_width=True)
        pct5 = (esperadas < 5).mean() * 100
        regra_ok = esperadas.min() >= 1 and pct5 <= 20
        (st.success if regra_ok else st.error)(
            f"Regra dos 5: menor esperada = {esperadas.min():.1f}; células com "
            f"esperada < 5 = {pct5:.0f}% → "
            f"{'condições atendidas ✅' if regra_ok else 'condições VIOLADAS — agrupe categorias ❌'}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Qui-quadrado", f"{chi2_obs:.4f}")
        c2.metric("Graus de liberdade", f"{gl_qq}")
        c3.metric("V de Cramér", f"{cramer_v(chi2_obs, tabela):.3f}")
        mostra_decisao(p_valor, ALPHA)
        fig_dist = grafico_distribuicao(chi2(df=gl_qq), chi2_obs, ALPHA,
                                        "unilateral à direita", "chi2")
        st.pyplot(fig_dist)
        plt.close(fig_dist)
        st.caption("Distribuição de referência: qui-quadrado (cauda superior).")

# ---------------------------------------------------------------------------
# Aba 5 — Planejamento amostral
# ---------------------------------------------------------------------------
with abas[4]:
    st.markdown("Calculadoras do tamanho mínimo de amostra (com correção para "
                f"população finita, N = {N:,}).")
    z_a2 = norm.ppf(1 - ALPHA / 2)

    with st.expander("📏 Média — teste bilateral com poder", expanded=True):
        c1, c2, c3 = st.columns(3)
        s_plan = c1.number_input("Desvio-padrão estimado (piloto)", 1.0, 500.0,
                                 55.82)
        delta = c2.number_input("Diferença mínima detectável (Δ)", 0.1, 100.0,
                                10.0)
        poder = c3.select_slider("Poder do teste", [0.80, 0.90, 0.95], 0.80,
                                 key="poder_media")
        n0 = ((z_a2 + norm.ppf(poder)) * s_plan / delta) ** 2
        n_fpc = int(np.ceil((n0 * N) / (n0 + N - 1)))
        st.metric("n mínimo (com correção)", n_fpc,
                  help=f"n inicial sem correção: {np.ceil(n0):.0f}")

    with st.expander("🎯 Proporção — margem de erro", expanded=True):
        c1, c2 = st.columns(2)
        E = c1.number_input("Margem de erro (E)", 0.005, 0.20, 0.04,
                            format="%.3f")
        p_plan = c2.number_input("Proporção esperada (0,5 = conservador)",
                                 0.05, 0.95, 0.50)
        n0 = (z_a2**2 * p_plan * (1 - p_plan)) / E**2
        n_fpc = int(np.ceil((n0 * N) / (n0 + N - 1)))
        st.metric("n mínimo (com correção)", n_fpc,
                  help=f"n inicial sem correção: {np.ceil(n0):.0f}")

    with st.expander("🧩 Qui-quadrado — tamanho do efeito (w de Cohen)",
                     expanded=True):
        c1, c2, c3 = st.columns(3)
        w = c1.select_slider("Efeito w", [0.10, 0.15, 0.30, 0.50], 0.15)
        gl_plan = c2.slider("Graus de liberdade da tabela", 1, 8, 3)
        frac = c3.slider("Fração de casos completos", 0.3, 1.0, 0.62)
        n0 = GofChisquarePower().solve_power(effect_size=w, alpha=ALPHA,
                                             power=0.80, n_bins=gl_plan + 1)
        n_aj = n0 / frac
        n_fpc = int(np.ceil((n_aj * N) / (n_aj + N - 1)))
        st.metric("n mínimo (ajustado por faltantes + correção)", n_fpc,
                  help=f"n sem ajustes: {np.ceil(n0):.0f}")

    st.info(f"O n do trabalho (752) é o máximo dos três critérios com os valores "
            f"padrão acima e α = 0,05. Ajuste o **n** na barra lateral para ver o "
            f"efeito nos testes — atualmente n = {n}.")

# ---------------------------------------------------------------------------
# Aba 6 — Validação populacional
# ---------------------------------------------------------------------------
with abas[5]:
    st.markdown("Como a população é totalmente conhecida (censo disponível), "
                "podemos comparar as estimativas amostrais com o gabarito.")
    xbar = amostra["training_hours"].mean()
    s = amostra["training_hours"].std(ddof=1)
    erro = t.ppf(1 - ALPHA / 2, df=n - 1) * s / np.sqrt(n)
    p_hat = amostra["target"].mean()
    li_p, ls_p = proportion_confint(int(amostra["target"].sum()), n,
                                    alpha=ALPHA, method="normal")
    mu_pop = pop["training_hours"].mean()
    p_pop = pop["target"].mean()
    tabela_val = pd.DataFrame({
        "Estimativa amostral": [f"{xbar:.2f}", f"{p_hat:.4f}"],
        f"IC {conf:.0%}": [f"({xbar - erro:.2f}; {xbar + erro:.2f})",
                           f"({li_p:.4f}; {ls_p:.4f})"],
        "Valor populacional": [f"{mu_pop:.2f}", f"{p_pop:.4f}"],
        "IC contém o valor?": [
            "✅" if xbar - erro <= mu_pop <= xbar + erro else "❌",
            "✅" if li_p <= p_pop <= ls_p else "❌",
        ],
    }, index=["Média de training_hours (h)", "Proporção que busca mudança"])
    st.dataframe(tabela_val, use_container_width=True)

    st.subheader("Taxa de busca por mudança por grupo — amostra × população")
    var_c = st.selectbox("Variável", VARS_CATEGORICAS, key="val_cat")
    comp = pd.DataFrame({
        "Amostra": amostra.groupby(var_c)["target"].mean().round(3),
        "População": pop.groupby(var_c)["target"].mean().round(3),
    })
    if var_c == "faixa_experiencia":
        comp = comp.reindex(ORDEM_EXPERIENCIA)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(comp, use_container_width=True)
    with c2:
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        comp.plot(kind="bar", ax=ax, color=["steelblue", "indianred"])
        ax.set_ylabel("Proporção que busca mudança")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
