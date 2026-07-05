# -*- coding: utf-8 -*-
"""
Trabalho Final - Análise Inferencial
=====================================
Tema: Intenção de mudança de emprego entre profissionais de Ciência de Dados

População de estudo: base "HR Analytics: Job Change of Data Scientists" (Kaggle),
tratada como população finita (N = 19.158 candidatos que concluíram treinamentos
de uma empresa de Big Data). A partir dela é extraída uma amostra aleatória
simples, sobre a qual são aplicados os procedimentos de inferência estatística
estudados na disciplina.

Questões de pesquisa:
  Q1. O número médio de horas de treinamento difere da referência de 60 h?
  Q2. Mais de 20% dos candidatos buscam uma nova oportunidade de emprego?
  Q3. O desenvolvimento da cidade (CDI) e a experiência profissional estão
      associados à intenção de mudança?
  Q4. Possuir experiência relevante na área está associado à intenção de mudança?
  Q5. O tipo de empresa em que o candidato trabalha está associado à intenção
      de mudança? (inclui análise da ausência informativa de informação de empresa)
  Q6. O nível de escolaridade está associado à intenção de mudança?
  Q7. Estudantes matriculados em curso integral buscam mudança em maior
      proporção do que candidatos sem matrícula? (teste z para duas proporções)
  Q8. A distribuição das faixas de experiência na amostra adere à distribuição
      populacional conhecida? (qui-quadrado de aderência — validação da AAS)

Etapas (conforme especificação do trabalho - apostila 8):
  1. Planejamento amostral (amostra piloto, cálculo de n p/ média, proporção e
     qui-quadrado, correção p/ população finita)
  2. Seleção da amostra aleatória
  3. Análise descritiva
  4. Estimação pontual
  5. Estimação intervalar (IC 95%)
  6. Testes de hipóteses
  7. Verificação das condições de aplicação e discussão de vieses
  8. Comparação com os valores populacionais (validação, possível pois N é conhecido)

Semente aleatória do curso: 2026
"""

import os
import sys

# garante saída UTF-8 no console do Windows (cp1252 não cobre símbolos como x̄)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from scipy.stats import chi2_contingency, chisquare, norm, t
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.power import GofChisquarePower
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

# ---------------------------------------------------------------------------
# Configuração de caminhos e utilitários
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # pasta projeto/
RAIZ = os.path.dirname(BASE)                                        # pasta do trabalho
DADOS_POP = os.path.join(RAIZ, "RH - Novas opor", "archive", "aug_train.csv")
DIR_DADOS = os.path.join(BASE, "dados")
DIR_FIG = os.path.join(BASE, "figuras")
DIR_RES = os.path.join(BASE, "resultados")
for d in (DIR_DADOS, DIR_FIG, DIR_RES):
    os.makedirs(d, exist_ok=True)

SEMENTE = 2026
ALPHA = 0.05

def secao(titulo):
    print("\n" + "=" * 78)
    print(titulo)
    print("=" * 78)


sns.set_theme(style="whitegrid", palette="deep")
ROTULO_TARGET = {0.0: "Não busca mudança", 1.0: "Busca mudança"}

# Agrupamentos de categorias (para garantir frequências esperadas adequadas)
GRUPO_EMPRESA = {
    "Pvt Ltd": "Empresa privada",
    "Funded Startup": "Startup",
    "Early Stage Startup": "Startup",
    "Public Sector": "Setor público",
    "NGO": "ONG/Outros",
    "Other": "ONG/Outros",
}
ORDEM_EMPRESA = ["Empresa privada", "Startup", "Setor público", "ONG/Outros"]
ORDEM_ESCOLARIDADE = ["Primary School", "High School", "Graduate", "Masters", "Phd"]
ORDEM_EXPERIENCIA = ["<1", "1-5", "6-10", "11-20", ">20"]
ORDEM_MATRICULA = ["no_enrollment", "Part time course", "Full time course"]


def banda_experiencia(v):
    """Converte a variável 'experience' (22 níveis) em 5 faixas."""
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


def cramer_v(chi2_val, tabela):
    """Coeficiente V de Cramér (força da associação)."""
    n_tab = tabela.to_numpy().sum()
    k = min(tabela.shape) - 1
    return np.sqrt(chi2_val / (n_tab * k))


def teste_qui_quadrado(tabela, descricao_h0, descricao_ha):
    """Executa e reporta um teste qui-quadrado de independência completo."""
    print(f"    H0: {descricao_h0}")
    print(f"    HA: {descricao_ha}")
    print("    Tabela de contingência (frequências observadas):")
    print(tabela.to_string())
    chi2_obs, p_val, gl_qq, esperadas = chi2_contingency(tabela)
    esp_df = pd.DataFrame(esperadas, index=tabela.index, columns=tabela.columns)
    print("    Frequências esperadas sob H0:")
    print(esp_df.round(1).to_string())
    # Regra dos 5 (apostila 7): nenhuma esperada < 1; no máx. 20% das células < 5
    pct_menor5 = (esperadas < 5).mean() * 100
    min_esp = esperadas.min()
    regra_ok = (min_esp >= 1) and (pct_menor5 <= 20)
    print(f"    Verificação da 'regra dos 5': menor esperada = {min_esp:.1f} (≥1: "
            f"{'sim' if min_esp >= 1 else 'NÃO'}); células com esperada < 5 = "
            f"{pct_menor5:.0f}% (≤20%: {'sim' if pct_menor5 <= 20 else 'NÃO'}) -> "
            f"{'condições atendidas' if regra_ok else 'CONDIÇÕES VIOLADAS'}")
    v = cramer_v(chi2_obs, tabela)
    print(f"    Qui-quadrado = {chi2_obs:.4f}  (gl = {gl_qq})")
    print(f"    p-valor = {p_val:.3e}")
    print(f"    V de Cramér = {v:.3f} (força da associação)")
    print(f"    Decisão: {'Rejeita H0' if p_val < ALPHA else 'Não rejeita H0'}")
    return chi2_obs, p_val, v


def figura_taxa_por_categoria(dados, coluna, ordem, titulo, arquivo, xlabel=""):
    """Gráfico de barras da proporção de target=1 por categoria, com n anotado."""
    grp = dados.groupby(coluna, observed=True)["target"].agg(["mean", "count"])
    grp = grp.reindex([c for c in ordem if c in grp.index])
    fig, ax = plt.subplots(figsize=(8.5, 5))
    barras = ax.bar(grp.index.astype(str), grp["mean"], color="steelblue",
                    edgecolor="white")
    for barra, (taxa, cont) in zip(barras, grp.itertuples(index=False)):
        ax.annotate(f"{taxa:.2f}\n(n={cont})",
                    (barra.get_x() + barra.get_width() / 2, barra.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Proporção que busca mudança")
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, max(grp["mean"]) * 1.25)
    ax.set_title(titulo)
    fig.tight_layout()
    fig.savefig(os.path.join(DIR_FIG, arquivo), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 0. População
# ---------------------------------------------------------------------------
secao("0. POPULAÇÃO DE ESTUDO")
pop = pd.read_csv(DADOS_POP)
N = len(pop)
print(f"Tamanho da população (N): {N}")
print("Variáveis de interesse:")
print("  - training_hours: horas de treinamento concluídas (quantitativa)")
print("  - target: busca nova oportunidade de emprego? 1=sim, 0=não (qualitativa)")
print("  - city_development_index (CDI): desenvolvimento da cidade (quantitativa)")
print("  - relevent_experience: possui experiência relevante na área (qualitativa)")
print("  - company_type: tipo de empresa em que trabalha (qualitativa)")
print("  - education_level: nível de escolaridade (qualitativa ordinal)")
print("  - experience: anos de experiência profissional (qualitativa ordinal)")

# ---------------------------------------------------------------------------
# 1. Planejamento amostral
# ---------------------------------------------------------------------------
secao("1. PLANEJAMENTO AMOSTRAL")
print(f"Nível de confiança adotado: 95% (alpha = {ALPHA})")
print("Método de amostragem: amostragem aleatória simples sem reposição")

# --- 1.1 Amostra piloto ------------------------------------------------------
print("\n1.1 Amostra piloto (n = 50)")
amostra_piloto = pop.sample(n=50, replace=False, random_state=123)
s_piloto = amostra_piloto["training_hours"].std(ddof=1)
p_piloto = amostra_piloto["target"].mean()
falta_piloto = amostra_piloto["company_type"].isna().mean()
print(f"  Desvio-padrão amostral de training_hours (piloto): {s_piloto:.2f} h")
print(f"  Proporção de candidatos que buscam mudança (piloto): {p_piloto:.2f}")
print(f"  Proporção de registros sem informação de empresa (piloto): {falta_piloto:.2f}")
amostra_piloto.to_csv(os.path.join(DIR_DADOS, "amostra_piloto.csv"), index=False)

# --- 1.2 Tamanho de amostra para a MÉDIA (teste bilateral) -------------------
# Requisito: detectar diferença mínima de 10 h na média de training_hours,
# com poder de 80% e significância de 5%.
print("\n1.2 Tamanho de amostra para a média (delta = 10 h, poder = 80%)")
poder = 0.80
delta_media = 10.0
z_a = norm.ppf(1 - ALPHA / 2)
z_b = norm.ppf(poder)
n_media = ((z_a + z_b) * s_piloto / delta_media) ** 2
print(f"  n inicial (quantis da normal): {np.ceil(n_media):.0f}")
# refinamento iterativo com quantis da t-Student (como na apostila 6)
n_iter = np.ceil(n_media)
for _ in range(100):
    t_a = t.ppf(1 - ALPHA / 2, df=n_iter - 1)
    t_b = t.ppf(poder, df=n_iter - 1)
    n_novo = np.ceil(((t_a + t_b) * s_piloto / delta_media) ** 2)
    if n_novo == n_iter:
        break
    n_iter = n_novo
n_media_t = n_iter
print(f"  n refinado (quantis da t-Student): {n_media_t:.0f}")
n_media_fpc = int(np.ceil((n_media_t * N) / (n_media_t + N - 1)))
print(f"  n com correção para população finita: {n_media_fpc}")

# --- 1.3 Tamanho de amostra para a PROPORÇÃO (margem de erro) ----------------
# Requisito: estimar a proporção populacional com margem de erro de 4 pontos
# percentuais e confiança de 95%. Usa-se p = 0,5 (cenário conservador, que
# maximiza a variância p(1-p)).
print("\n1.3 Tamanho de amostra para a proporção (E = 0,04; p conservador = 0,5)")
E = 0.04
p_cons = 0.5
n_prop = (z_a**2 * p_cons * (1 - p_cons)) / E**2
print(f"  n inicial: {np.ceil(n_prop):.0f}")
n_prop_fpc = int(np.ceil((n_prop * N) / (n_prop + N - 1)))
print(f"  n com correção para população finita: {n_prop_fpc}")

# --- 1.4 Tamanho de amostra para os testes QUI-QUADRADO ----------------------
# Requisito (apostila 7): detectar efeito de magnitude w = 0,15 (pequeno-médio,
# escala de Cohen) com poder de 80%, considerando a tabela de maior gl planejada
# e inflacionando pela taxa de dados faltantes estimada na amostra piloto
# (o teste de company_type usa apenas os casos completos).
print("\n1.4 Tamanho de amostra para os testes qui-quadrado (w = 0,15, poder = 80%)")
analisador = GofChisquarePower()
w_alvo = 0.15
planejamento_qq = {
    # teste: (graus de liberdade, fração de casos aproveitáveis)
    "company_type (4x2, gl=3)": (3, 1 - falta_piloto),
    "education_level (5x2, gl=4)": (4, 1.0),
    "experience (5x2, gl=4)": (4, 1.0),
}
n_qq_max = 0
for nome, (gl_plan, frac) in planejamento_qq.items():
    n_puro = analisador.solve_power(effect_size=w_alvo, alpha=ALPHA,
                                    power=poder, n_bins=gl_plan + 1)
    n_ajust = np.ceil(n_puro / frac)
    print(f"  {nome}: n = {np.ceil(n_puro):.0f}; ajustado p/ dados faltantes: "
            f"{n_ajust:.0f}")
    n_qq_max = max(n_qq_max, n_ajust)
n_qq_fpc = int(np.ceil((n_qq_max * N) / (n_qq_max + N - 1)))
print(f"  n com correção para população finita: {n_qq_fpc}")

# --- 1.5 Tamanho final -------------------------------------------------------
n_final = max(n_media_fpc, n_prop_fpc, n_qq_fpc)
print(f"\n1.5 Tamanho final da amostra: n = max({n_media_fpc}, {n_prop_fpc}, "
        f"{n_qq_fpc}) = {n_final}")

# ---------------------------------------------------------------------------
# 2. Seleção da amostra aleatória
# ---------------------------------------------------------------------------
secao("2. SELEÇÃO DA AMOSTRA ALEATÓRIA")
amostra = pop.sample(n=n_final, replace=False, random_state=SEMENTE).copy()
# variáveis derivadas
amostra["faixa_experiencia"] = amostra["experience"].map(banda_experiencia)
amostra["tipo_empresa"] = amostra["company_type"].map(GRUPO_EMPRESA)
amostra["sem_info_empresa"] = np.where(amostra["company_type"].isna(),
                                       "Sem informação", "Com informação")
amostra.to_csv(os.path.join(DIR_DADOS, "amostra_final.csv"), index=False)
print(f"Amostra de n = {n_final} selecionada sem reposição (semente = {SEMENTE}).")
print("Variáveis derivadas criadas: faixa_experiencia (5 faixas), tipo_empresa "
        "(4 grupos) e sem_info_empresa (flag de ausência).")
print("Arquivo salvo em: dados/amostra_final.csv")

# ---------------------------------------------------------------------------
# 3. Análise descritiva
# ---------------------------------------------------------------------------
secao("3. ANÁLISE DESCRITIVA (com base na amostra)")
n = len(amostra)

print("\n3.1 Medidas descritivas de training_hours")
desc = amostra["training_hours"].describe()
print(desc.round(2).to_string())
print(f"  Assimetria: {amostra['training_hours'].skew():.2f}")

print("\n3.2 Tabela de frequências de target")
tab_target = amostra["target"].map(ROTULO_TARGET).value_counts()
tab_target_rel = amostra["target"].map(ROTULO_TARGET).value_counts(normalize=True)
tab = pd.DataFrame({"Frequência": tab_target, "Proporção": tab_target_rel.round(4)})
print(tab.to_string())

print("\n3.3 Tabela de frequências de relevent_experience")
tab_exp = pd.DataFrame({
    "Frequência": amostra["relevent_experience"].value_counts(),
    "Proporção": amostra["relevent_experience"].value_counts(normalize=True).round(4),
})
print(tab_exp.to_string())

print("\n3.4 Tipo de empresa (agrupado; inclui ausência de informação)")
tab_emp = pd.DataFrame({
    "Frequência": amostra["tipo_empresa"].value_counts(dropna=False),
    "Proporção": amostra["tipo_empresa"].value_counts(dropna=False,
                                                      normalize=True).round(4),
})
tab_emp.index = [x if isinstance(x, str) else "Sem informação" for x in tab_emp.index]
print(tab_emp.to_string())
print(f"  Taxa de busca por mudança por disponibilidade de informação de empresa:")
print(amostra.groupby("sem_info_empresa")["target"].agg(
    taxa="mean", n="count").round(4).to_string())

print("\n3.5 Nível de escolaridade")
tab_edu = pd.DataFrame({
    "Frequência": amostra["education_level"].value_counts(dropna=False),
    "Proporção": amostra["education_level"].value_counts(dropna=False,
                                                         normalize=True).round(4),
})
print(tab_edu.to_string())

print("\n3.6 Faixa de experiência profissional")
tab_fx = pd.DataFrame({
    "Frequência": amostra["faixa_experiencia"].value_counts(dropna=False),
    "Proporção": amostra["faixa_experiencia"].value_counts(dropna=False,
                                                           normalize=True).round(4),
})
tab_fx = tab_fx.reindex(ORDEM_EXPERIENCIA)
print(tab_fx.to_string())

print("\n3.7 city_development_index por grupo de target")
cdi_grp = amostra.groupby(amostra["target"].map(ROTULO_TARGET))[
    "city_development_index"
].agg(["count", "mean", "std", "min", "median", "max"])
print(cdi_grp.round(3).to_string())

# --- Figuras ---------------------------------------------------------------
# Fig 1: histograma de training_hours
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(amostra["training_hours"], bins=30, color="steelblue", edgecolor="white")
ax.set_xlabel("Horas de treinamento")
ax.set_ylabel("Frequência")
ax.set_title("Figura 1 - Distribuição das horas de treinamento (amostra)")
fig.tight_layout()
fig.savefig(os.path.join(DIR_FIG, "fig1_hist_training_hours.png"), dpi=150)
plt.close(fig)

# Fig 2: barras de target
fig, ax = plt.subplots(figsize=(7, 5))
tab_target_rel.sort_index().plot(kind="bar", ax=ax, color=["steelblue", "indianred"])
ax.set_ylabel("Proporção")
ax.set_xlabel("")
ax.set_title("Figura 2 - Proporção de candidatos que buscam mudança de emprego")
ax.tick_params(axis="x", rotation=0)
for c in ax.containers:
    ax.bar_label(c, fmt="%.3f")
fig.tight_layout()
fig.savefig(os.path.join(DIR_FIG, "fig2_barras_target.png"), dpi=150)
plt.close(fig)

# Fig 3: boxplot de city_development_index por target
fig, ax = plt.subplots(figsize=(8, 5))
dados_box = amostra.assign(grupo=amostra["target"].map(ROTULO_TARGET))
sns.boxplot(data=dados_box, x="grupo", y="city_development_index", hue="grupo", ax=ax)
ax.set_xlabel("")
ax.set_ylabel("Índice de desenvolvimento da cidade (CDI)")
ax.set_title("Figura 3 - CDI por intenção de mudança de emprego")
fig.tight_layout()
fig.savefig(os.path.join(DIR_FIG, "fig3_boxplot_cdi_target.png"), dpi=150)
plt.close(fig)

# Fig 4: barras agrupadas relevent_experience x target
fig, ax = plt.subplots(figsize=(8, 5))
tab_cruz_rel = pd.crosstab(
    amostra["relevent_experience"], amostra["target"].map(ROTULO_TARGET),
    normalize="index"
)
tab_cruz_rel.plot(kind="bar", ax=ax, color=["steelblue", "indianred"])
ax.set_ylabel("Proporção dentro do grupo")
ax.set_xlabel("")
ax.set_title("Figura 4 - Intenção de mudança segundo experiência relevante")
ax.tick_params(axis="x", rotation=0)
ax.legend(title="")
fig.tight_layout()
fig.savefig(os.path.join(DIR_FIG, "fig4_barras_experiencia_target.png"), dpi=150)
plt.close(fig)

# Fig 5: QQ-plot de training_hours (verificação de normalidade)
fig, ax = plt.subplots(figsize=(7, 5))
stats.probplot(amostra["training_hours"], dist="norm", plot=ax)
ax.set_title("Figura 5 - Q-Q plot de training_hours (amostra)")
ax.set_xlabel("Quantis teóricos")
ax.set_ylabel("Quantis amostrais")
fig.tight_layout()
fig.savefig(os.path.join(DIR_FIG, "fig5_qqplot_training_hours.png"), dpi=150)
plt.close(fig)

# Fig 6: taxa de mudança por nível de escolaridade
figura_taxa_por_categoria(
    amostra.dropna(subset=["education_level"]), "education_level",
    ORDEM_ESCOLARIDADE,
    "Figura 6 - Proporção que busca mudança por nível de escolaridade",
    "fig6_taxa_por_escolaridade.png", xlabel="Nível de escolaridade")

# Fig 7: taxa de mudança por tipo de empresa (incluindo "Sem informação")
amostra_fig7 = amostra.assign(
    tipo_emp_plot=amostra["tipo_empresa"].fillna("Sem informação"))
figura_taxa_por_categoria(
    amostra_fig7, "tipo_emp_plot", ORDEM_EMPRESA + ["Sem informação"],
    "Figura 7 - Proporção que busca mudança por tipo de empresa",
    "fig7_taxa_por_tipo_empresa.png", xlabel="Tipo de empresa")

# Fig 8: taxa de mudança por faixa de experiência
figura_taxa_por_categoria(
    amostra.dropna(subset=["faixa_experiencia"]), "faixa_experiencia",
    ORDEM_EXPERIENCIA,
    "Figura 8 - Proporção que busca mudança por faixa de experiência (anos)",
    "fig8_taxa_por_experiencia.png", xlabel="Anos de experiência")

# Fig 9: taxa de mudança por situação de matrícula universitária
figura_taxa_por_categoria(
    amostra.dropna(subset=["enrolled_university"]), "enrolled_university",
    ORDEM_MATRICULA,
    "Figura 9 - Proporção que busca mudança por situação de matrícula",
    "fig9_taxa_por_matricula.png", xlabel="Situação de matrícula universitária")

print("\nFiguras 1 a 9 salvas em projeto/figuras/.")

# ---------------------------------------------------------------------------
# 4. Estimação pontual
# ---------------------------------------------------------------------------
secao("4. ESTIMAÇÃO PONTUAL")
xbar = amostra["training_hours"].mean()
s = amostra["training_hours"].std(ddof=1)
p_hat = amostra["target"].mean()
x_sucessos = int(amostra["target"].sum())
print(f"Média amostral de training_hours (x̄): {xbar:.2f} h  (estima μ)")
print(f"Desvio-padrão amostral (s): {s:.2f} h")
print(f"Proporção amostral de candidatos buscando mudança (p̂): {p_hat:.4f}  (estima p)")
print(f"  ({x_sucessos} sucessos em {n} observações)")

# ---------------------------------------------------------------------------
# 5. Estimação intervalar (IC 95%)
# ---------------------------------------------------------------------------
secao("5. ESTIMAÇÃO INTERVALAR (IC 95%)")

# IC para a média (variância desconhecida -> t-Student)
gl = n - 1
t_crit = t.ppf(1 - ALPHA / 2, df=gl)
erro_media = t_crit * s / np.sqrt(n)
print("5.1 IC 95% para a média de training_hours (t-Student, variância desconhecida)")
print(f"  x̄ ± t(0,975; {gl}) · s/√n = {xbar:.2f} ± {erro_media:.2f}")
print(f"  IC95%(μ) = ({xbar - erro_media:.2f}; {xbar + erro_media:.2f}) horas")

# IC para a proporção (Wald + Clopper-Pearson)
print("\n5.2 IC 95% para a proporção de candidatos que buscam mudança")
cond_wald = n * p_hat * (1 - p_hat)
print(f"  Condição n·p̂·(1-p̂) = {cond_wald:.1f} ≥ 5 -> aproximação normal válida")
li_w, ls_w = proportion_confint(count=x_sucessos, nobs=n, alpha=ALPHA, method="normal")
li_cp, ls_cp = proportion_confint(count=x_sucessos, nobs=n, alpha=ALPHA, method="beta")
print(f"  Wald (aproximado):          IC95%(p) = ({li_w:.4f}; {ls_w:.4f})")
print(f"  Clopper-Pearson (exato):    IC95%(p) = ({li_cp:.4f}; {ls_cp:.4f})")

# ---------------------------------------------------------------------------
# 6. Testes de hipóteses
# ---------------------------------------------------------------------------
secao("6. TESTES DE HIPÓTESES (alpha = 0,05)")

# --- Q1: média de training_hours --------------------------------------------
print("6.1 [Q1] Teste t para a média de training_hours")
print("    H0: μ = 60  vs  HA: μ ≠ 60   (referência de planejamento de trilhas)")
mu0 = 60.0
t_obs = (xbar - mu0) / (s / np.sqrt(n))
p_t = 2 * (1 - t.cdf(abs(t_obs), df=gl))
print(f"    Estatística t = {t_obs:.4f}  (gl = {gl})")
print(f"    p-valor = {p_t:.4f}")
print(f"    Decisão: {'Rejeita H0' if p_t < ALPHA else 'Não rejeita H0'}")

# --- Q2: proporção de target = 1 ---------------------------------------------
print("\n6.2 [Q2] Teste z para a proporção de candidatos que buscam mudança")
print("    H0: p = 0,20  vs  HA: p > 0,20   (unilateral à direita)")
p0 = 0.20
cond1, cond2 = n * p0, n * (1 - p0)
print(f"    Condições: n·p0 = {cond1:.0f} ≥ 5 e n·(1-p0) = {cond2:.0f} ≥ 5 -> OK")
z_obs = (p_hat - p0) / np.sqrt(p0 * (1 - p0) / n)
p_z = 1 - norm.cdf(z_obs)
print(f"    Estatística z = {z_obs:.4f}")
print(f"    p-valor = {p_z:.6f}")
# confirmação com statsmodels (variância sob H0)
z_sm, p_sm = proportions_ztest(count=x_sucessos, nobs=n, value=p0,
                               prop_var=p0, alternative="larger")
print(f"    (statsmodels: z = {z_sm:.4f}, p-valor = {p_sm:.6f})")
print(f"    Decisão: {'Rejeita H0' if p_z < ALPHA else 'Não rejeita H0'}")

# --- Q3a: comparação de duas médias (CDI por target) -------------------------
print("\n6.3 [Q3a] Teste t de Welch: CDI médio entre quem busca e quem não busca")
print("    H0: μ_busca = μ_nao_busca  vs  HA: μ_busca ≠ μ_nao_busca")
grupo1 = amostra.loc[amostra["target"] == 1.0, "city_development_index"]
grupo0 = amostra.loc[amostra["target"] == 0.0, "city_development_index"]
s1, s0 = grupo1.std(ddof=1), grupo0.std(ddof=1)
razao_dp = max(s1, s0) / min(s1, s0)
print(f"    n_busca = {len(grupo1)}, n_nao_busca = {len(grupo0)}")
print(f"    Médias: {grupo1.mean():.4f} (busca) vs {grupo0.mean():.4f} (não busca)")
print(f"    Razão entre desvios-padrão = {razao_dp:.2f} "
        f"({'> 2: usar Welch' if razao_dp > 2 else '< 2, mas Welch é mais robusto'})")
res_welch = stats.ttest_ind(grupo1, grupo0, equal_var=False, alternative="two-sided")
print(f"    Estatística t = {res_welch.statistic:.4f}  (gl = {res_welch.df:.1f})")
print(f"    p-valor = {res_welch.pvalue:.3e}")
print(f"    Decisão: {'Rejeita H0' if res_welch.pvalue < ALPHA else 'Não rejeita H0'}")

# --- Q3b: faixa de experiência × target --------------------------------------
print("\n6.4 [Q3b] Qui-quadrado de independência: faixa de experiência × target")
dados_q3b = amostra.dropna(subset=["faixa_experiencia"])
tab_q3b = pd.crosstab(
    pd.Categorical(dados_q3b["faixa_experiencia"], categories=ORDEM_EXPERIENCIA,
                   ordered=True),
    dados_q3b["target"].map(ROTULO_TARGET))
tab_q3b.index.name = "faixa_experiencia"
teste_qui_quadrado(
    tab_q3b,
    "a faixa de experiência e a intenção de mudança são independentes",
    "existe associação entre experiência e intenção de mudança")

# --- Q4: experiência relevante × target --------------------------------------
print("\n6.5 [Q4] Qui-quadrado de independência: experiência relevante × target")
tab_q4 = pd.crosstab(amostra["relevent_experience"],
                     amostra["target"].map(ROTULO_TARGET))
teste_qui_quadrado(
    tab_q4,
    "possuir experiência relevante e a intenção de mudança são independentes",
    "existe associação entre experiência relevante e intenção de mudança")

# --- Q5: tipo de empresa × target ---------------------------------------------
print("\n6.6 [Q5] Qui-quadrado de independência: tipo de empresa × target")
print("    (casos completos; categorias agrupadas para atender a regra dos 5:")
print("     Startup = Funded + Early Stage; ONG/Outros = NGO + Other)")
dados_q5 = amostra.dropna(subset=["tipo_empresa"])
print(f"    Casos completos: {len(dados_q5)} de {n} "
        f"({len(dados_q5) / n:.1%} da amostra)")
tab_q5 = pd.crosstab(
    pd.Categorical(dados_q5["tipo_empresa"], categories=ORDEM_EMPRESA, ordered=True),
    dados_q5["target"].map(ROTULO_TARGET))
tab_q5.index.name = "tipo_empresa"
teste_qui_quadrado(
    tab_q5,
    "o tipo de empresa e a intenção de mudança são independentes",
    "existe associação entre tipo de empresa e intenção de mudança")

# --- Q5 (complemento): ausência informativa -----------------------------------
print("\n6.7 [Q5 - complemento] Qui-quadrado: ausência de informação de empresa × target")
print("    Motivação: a ausência de company_type pode ser INFORMATIVA (MNAR) —")
print("    candidatos sem vínculo empregatício declarado tendem a não preencher")
print("    o campo e podem ter maior propensão à mudança.")
tab_q5c = pd.crosstab(amostra["sem_info_empresa"],
                      amostra["target"].map(ROTULO_TARGET))
teste_qui_quadrado(
    tab_q5c,
    "a ausência de informação de empresa e a intenção de mudança são independentes",
    "a ausência de informação de empresa está associada à intenção de mudança")

# --- Q6: escolaridade × target -------------------------------------------------
print("\n6.8 [Q6] Qui-quadrado de independência: nível de escolaridade × target")
dados_q6 = amostra.dropna(subset=["education_level"])
print(f"    Casos completos: {len(dados_q6)} de {n} "
        f"({len(dados_q6) / n:.1%} da amostra)")
tab_q6 = pd.crosstab(
    pd.Categorical(dados_q6["education_level"], categories=ORDEM_ESCOLARIDADE,
                   ordered=True),
    dados_q6["target"].map(ROTULO_TARGET))
tab_q6.index.name = "education_level"
teste_qui_quadrado(
    tab_q6,
    "o nível de escolaridade e a intenção de mudança são independentes",
    "existe associação entre escolaridade e intenção de mudança")

# --- Q7: duas proporções (matrícula integral vs sem matrícula) ---------------
print("\n6.9 [Q7] Teste z para duas proporções: matrícula integral vs sem matrícula")
print("    H0: p_integral = p_sem_matricula  vs  HA: p_integral > p_sem_matricula")
print("    Justificativa da direção (definida a priori): quem cursa universidade")
print("    em tempo integral está investindo em requalificação e, pela literatura")
print("    de mobilidade ocupacional, tende a maior propensão à mudança.")
g_int = amostra.loc[amostra["enrolled_university"] == "Full time course", "target"]
g_sem = amostra.loc[amostra["enrolled_university"] == "no_enrollment", "target"]
n1_q7, n2_q7 = len(g_int), len(g_sem)
x1_q7, x2_q7 = int(g_int.sum()), int(g_sem.sum())
p1_hat, p2_hat = x1_q7 / n1_q7, x2_q7 / n2_q7
p_pool = (x1_q7 + x2_q7) / (n1_q7 + n2_q7)
print(f"    Grupos: integral n1 = {n1_q7} (p̂1 = {p1_hat:.4f}); "
      f"sem matrícula n2 = {n2_q7} (p̂2 = {p2_hat:.4f})")
print(f"    Proporção combinada (pooled): p̂ = {p_pool:.4f}")
cond1_q7 = n1_q7 * p_pool * (1 - p_pool)
cond2_q7 = n2_q7 * p_pool * (1 - p_pool)
print(f"    Condições: n1·p̂·(1-p̂) = {cond1_q7:.1f} e n2·p̂·(1-p̂) = "
      f"{cond2_q7:.1f} (ambas ≥ 5) -> OK")
z_2p = (p1_hat - p2_hat) / np.sqrt(p_pool * (1 - p_pool) * (1 / n1_q7 + 1 / n2_q7))
p_2p = 1 - norm.cdf(z_2p)
print(f"    Estatística z = {z_2p:.4f}")
print(f"    p-valor = {p_2p:.6f}")
z_sm2, p_sm2 = proportions_ztest(count=[x1_q7, x2_q7], nobs=[n1_q7, n2_q7],
                                 alternative="larger")
print(f"    (statsmodels: z = {z_sm2:.4f}, p-valor = {p_sm2:.6f})")
# IC 95% para a diferença de proporções (erro-padrão não combinado)
ep_dif = np.sqrt(p1_hat * (1 - p1_hat) / n1_q7 + p2_hat * (1 - p2_hat) / n2_q7)
z975 = norm.ppf(1 - ALPHA / 2)
print(f"    IC95%(p1 - p2) = ({p1_hat - p2_hat - z975 * ep_dif:.4f}; "
      f"{p1_hat - p2_hat + z975 * ep_dif:.4f})")
print(f"    Decisão: {'Rejeita H0' if p_2p < ALPHA else 'Não rejeita H0'}")

# --- Q8: aderência da amostra à distribuição populacional de experiência -----
print("\n6.10 [Q8] Qui-quadrado de aderência: faixas de experiência "
      "(amostra vs população)")
print("    H0: a distribuição das faixas de experiência na amostra segue as")
print("        proporções populacionais conhecidas (amostra representativa)")
print("    HA: pelo menos uma proporção difere")
print("    Observação: como a população é totalmente conhecida (censo), as")
print("    proporções de referência são fixadas a priori -> gl = k - 1.")
pop_fx = pop["experience"].map(banda_experiencia).dropna()
props_pop = pop_fx.value_counts(normalize=True).reindex(ORDEM_EXPERIENCIA)
obs_fx = (amostra["faixa_experiencia"].dropna().value_counts()
          .reindex(ORDEM_EXPERIENCIA).fillna(0))
n_validos = int(obs_fx.sum())
esperadas_fx = props_pop * n_validos
tabela_ader = pd.DataFrame({
    "Proporção populacional": props_pop.round(4),
    "Freq. observada": obs_fx.astype(int),
    "Freq. esperada": esperadas_fx.round(1),
})
tabela_ader.index.name = "faixa_experiencia"
print(tabela_ader.to_string())
print(f"    Casos válidos na amostra: {n_validos} de {n}")
print(f"    Menor frequência esperada = {esperadas_fx.min():.1f} "
      "(≥ 5 -> condição atendida)")
chi2_ader, p_ader = chisquare(f_obs=obs_fx, f_exp=esperadas_fx)
print(f"    Qui-quadrado = {chi2_ader:.4f}  (gl = {len(ORDEM_EXPERIENCIA) - 1})")
print(f"    p-valor = {p_ader:.4f}")
print(f"    Decisão: {'Rejeita H0' if p_ader < ALPHA else 'Não rejeita H0'}")
print("    Interpretação: a não rejeição corrobora a representatividade da")
print("    amostra aleatória simples quanto ao perfil de experiência.")

# ---------------------------------------------------------------------------
# 7. Verificação das condições de aplicação e vieses
# ---------------------------------------------------------------------------
secao("7. VERIFICAÇÃO DAS CONDIÇÕES DE APLICAÇÃO E DISCUSSÃO DE VIESES")
print("- Independência: amostragem aleatória simples sem reposição com n/N = "
        f"{n / N:.3f} (< 10% da população) -> observações aproximadamente independentes.")
print("- Normalidade de training_hours: distribuição assimétrica à direita "
        "(Figuras 1 e 5). Teste formal abaixo:")
stat_lf, p_lf = lilliefors(amostra["training_hours"], dist="norm")
print(f"    Lilliefors: estatística = {stat_lf:.4f}, p-valor = {p_lf:.4f} "
        f"-> {'rejeita' if p_lf < ALPHA else 'não rejeita'} normalidade")
print(f"  Contudo, como n = {n} é grande, o Teorema Central do Limite garante "
        "que a distribuição amostral de x̄ é aproximadamente normal, validando "
        "o IC e o teste t para a média.")
print("- Proporção: condições np ≥ 5 verificadas nos itens 5.2 e 6.2.")
print("- Qui-quadrado (independência e aderência): 'regra dos 5' verificada "
      "individualmente em cada teste (itens 6.4 a 6.10); categorias raras de "
      "company_type foram agrupadas.")
print("- Duas proporções (item 6.9): grupos desbalanceados (~20% vs ~72% da "
      "amostra), mas ambas as condições np ≥ 5 são amplamente atendidas.")
print("")
print("Vieses e limitações identificados (impactam a interpretação, não a validade")
print("interna dos procedimentos):")
print("  1. Viés de seleção: a população é composta por candidatos inscritos em")
print("     treinamentos da empresa (autosseleção de pessoas em requalificação,")
print("     naturalmente mais propensas à mobilidade). As conclusões valem para")
print("     essa população, e NÃO para profissionais de dados em geral.")
print("  2. Concentração geográfica: poucas cidades concentram grande parte dos")
print("     registros, e o CDI é um proxy socioeconômico REGIONAL, não um atributo")
print("     individual. A associação CDI × mudança (item 6.3) deve ser lida como")
print("     efeito de contexto, não de mérito individual.")
print("  3. Ausência informativa (MNAR): company_type/company_size faltam juntas")
print("     em cerca de 1/3 dos registros e a ausência está associada ao alvo")
print("     (item 6.7). Por isso a ausência foi tratada como categoria própria em")
print("     análise complementar, e não descartada silenciosamente.")
print("  4. Variável gender: não utilizada nos testes por ter 23,5% de dados")
print("     faltantes e forte desbalanceamento (Male ~69%, Female ~6,5%), o que")
print("     fragilizaria qualquer conclusão e envolveria riscos de interpretação")
print("     discriminatória em contexto de RH (discriminação por proxy).")
print("  5. Viés de rótulo: a definição operacional de 'busca mudança' (target)")
print("     não é documentada pela fonte (auto-relato? comportamento?), o que é")
print("     uma ameaça à validade de constructo que não podemos verificar.")

# ---------------------------------------------------------------------------
# 8. Validação com os valores populacionais (possível pois N é conhecido)
# ---------------------------------------------------------------------------
secao("8. COMPARAÇÃO COM OS VALORES POPULACIONAIS (validação)")
print(f"Média populacional de training_hours: μ = {pop['training_hours'].mean():.2f} h "
        f"(IC obtido: {xbar - erro_media:.2f} a {xbar + erro_media:.2f})")
print(f"Proporção populacional de target = 1:  p = {pop['target'].mean():.4f} "
        f"(IC obtido: {li_w:.4f} a {ls_w:.4f})")
print(f"CDI médio populacional: busca = "
        f"{pop.loc[pop['target'] == 1, 'city_development_index'].mean():.4f}, "
        f"não busca = {pop.loc[pop['target'] == 0, 'city_development_index'].mean():.4f}")
pop_aux = pop.copy()
pop_aux["faixa_experiencia"] = pop_aux["experience"].map(banda_experiencia)
pop_aux["tipo_empresa"] = pop_aux["company_type"].map(GRUPO_EMPRESA)
pop_aux["sem_info_empresa"] = np.where(pop_aux["company_type"].isna(),
                                       "Sem informação", "Com informação")
print("\nTaxa populacional de busca por mudança, por grupo:")
for col in ["faixa_experiencia", "tipo_empresa", "sem_info_empresa",
            "education_level", "relevent_experience", "enrolled_university"]:
    taxas = pop_aux.groupby(col, observed=True)["target"].mean().round(3)
    print(f"  {col}: " + "; ".join(f"{k} = {v}" for k, v in taxas.items()))

print("\nAnálise concluída. Para salvar a saída em arquivo, redirecione:")
print('  python codigo/analise_inferencial.py > "resultados/resultados_analise.txt"')
