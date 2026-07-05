# Trabalho Final — Análise Inferencial

**Tema:** Intenção de mudança de emprego entre profissionais de Ciência de Dados

**População de estudo:** base *HR Analytics: Job Change of Data Scientists* (Kaggle),
com N = 19.158 candidatos que concluíram treinamentos oferecidos por uma empresa de
Big Data. A base é tratada como população finita, da qual se extrai uma amostra
aleatória simples para aplicação dos métodos de inferência estatística.

## Estrutura

```
projeto/
├── app/
│   └── app_streamlit.py         # app interativo (Streamlit)
├── codigo/
│   └── analise_inferencial.py   # script completo da análise (Python)
├── dados/
│   ├── amostra_piloto.csv       # amostra piloto (n = 50) usada no planejamento
│   └── amostra_final.csv        # amostra aleatória usada em todas as análises
├── figuras/                     # gráficos gerados (Figuras 1 a 9)
├── resultados/
│   └── resultados_analise.txt   # saída numérica completa da análise
└── relatorio/                   # relatório final (PDF)
```

## Etapas implementadas (conforme especificação do trabalho)

1. **Planejamento amostral** — nível de confiança de 95%; amostra piloto (n = 50);
   tamanho amostral calculado para três requisitos e tomado o máximo: média
   (Δ = 10 h, poder 80% → 244), proporção (margem de 4 p.p., p = 0,5 → 583) e
   testes qui-quadrado (w = 0,15, poder 80%, ajustado por dados faltantes → 752);
   correção para população finita em todos. **n final = 752**.
2. **Coleta** — amostragem aleatória simples sem reposição, semente 2026.
3. **Análise descritiva** — tabelas de frequência, medidas descritivas e 9 figuras.
4. **Estimação pontual** — média de `training_hours` e proporção de `target = 1`.
5. **Estimação intervalar** — IC 95% t-Student para a média; IC 95% Wald e
   Clopper-Pearson para a proporção.
6. **Testes de hipóteses** (Q1 a Q8)
   - Q1: teste t para a média: H0: μ = 60 h vs HA: μ ≠ 60 h;
   - Q2: teste z para a proporção: H0: p = 0,20 vs HA: p > 0,20;
   - Q3a: teste t de Welch: CDI médio entre quem busca e quem não busca mudança;
   - Q3b: qui-quadrado: faixa de experiência × intenção de mudança;
   - Q4: qui-quadrado: experiência relevante × intenção de mudança;
   - Q5: qui-quadrado: tipo de empresa (4 grupos) × intenção de mudança +
     complemento de ausência informativa (sem_info_empresa × target);
   - Q6: qui-quadrado: nível de escolaridade × intenção de mudança;
   - Q7: teste z para **duas proporções**: matrícula universitária integral vs
     sem matrícula (proporção combinada/pooled, IC 95% para p₁ − p₂);
   - Q8: qui-quadrado de **aderência**: distribuição das faixas de experiência
     da amostra vs proporções populacionais conhecidas (validação da
     representatividade da AAS).
7. **Verificação das condições e vieses** — independência, normalidade (Q-Q plot e
   Lilliefors), TCL, condições np ≥ 5, "regra dos 5" (com agrupamento de categorias
   raras) e discussão de vieses (seleção, geografia/CDI como proxy, ausência
   informativa MNAR, gender, rótulo).
8. **Validação** — comparação dos resultados amostrais com os valores populacionais
   verdadeiros (possível porque a população é conhecida).

## Como executar

### Script da análise (gera figuras, amostras e resultados)

```bash
pip install pandas numpy scipy statsmodels matplotlib seaborn
python codigo/analise_inferencial.py
# para salvar a saída numérica em arquivo:
python codigo/analise_inferencial.py > "resultados/resultados_analise.txt"
```

### App interativo (Streamlit)

```bash
pip install streamlit
streamlit run projeto/app/app_streamlit.py   # a partir da pasta raiz do trabalho
```

No app é possível manipular o **tamanho da amostra**, a **semente**, o **nível
de confiança**, o **tipo de teste** (t para média, z para proporção, Welch,
qui-quadrado) e suas hipóteses alternativas, visualizando tabelas, métricas,
a distribuição de referência com a região crítica e a validação contra os
valores populacionais. Há também calculadoras de planejamento amostral.

Toda a análise é reprodutível: as sementes aleatórias estão fixadas
(piloto: 123; amostra final: 2026).

## Perguntas adicionais documentadas (não implementadas no script)

Sugestões com sinal já verificado na população (censo disponível), úteis como
extensão futura ou para a defesa oral. **As três podem ser exploradas
interativamente no app**, na aba "Testes de hipóteses":

1. **Horas de treinamento diferem entre quem busca e quem não busca mudança?**
   (t de Welch, variável `training_hours` por `target`) — na população a
   diferença é pequena (63,1 h vs 66,1 h, com dp ≈ 60 h); o resultado esperado
   é a **não rejeição** de H0, formalizando que `training_hours` carrega pouco
   sinal sobre o desfecho (V de Cramér ≈ 0,02 na análise de vieses).
2. **Quem nunca trocou de emprego busca mais mudança?** (qui-quadrado,
   `last_new_job` × `target`) — gradiente populacional monotônico:
   "never" 30,1% → "1 ano" 26,4% → ">4 anos" 18,2%.
3. **O porte da empresa está associado à mudança?** (qui-quadrado,
   `company_size` × `target`) — variação moderada entre os informados
   (15% a 23%) e o mesmo fenômeno de ausência informativa da Q5
   (40,6% de faltantes com taxa de 40,6% de busca), reforçando o achado
   central do trabalho.

Descartadas deliberadamente: `major_discipline` (86% dos casos são STEM —
pouca variação para detectar) e `gender` (23,5% de dados faltantes e forte
desbalanceamento; ver a seção de vieses do relatório).
