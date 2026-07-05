# Intenção de Mudança de Emprego entre Profissionais de Ciência de Dados: uma Análise Inferencial

**Disciplina:** Análise Inferencial — Trabalho Final
**Aluno:** Lucas Ferreira da Silva — 567414

## 1. Introdução

A retenção de talentos é um dos principais desafios enfrentados por empresas de
tecnologia. No campo da Ciência de Dados, em que a demanda por profissionais
qualificados supera a oferta, compreender quais fatores estão associados à
intenção de mudar de emprego permite às organizações desenhar políticas de
retenção mais eficazes e planejar melhor seus programas de capacitação.

Este trabalho aplica os métodos de inferência estatística estudados na
disciplina a um problema real de gestão de pessoas. A população de estudo é
composta por **N = 19.158 candidatos** que concluíram treinamentos oferecidos
por uma empresa de Big Data e Ciência de Dados, cujos registros estão
disponíveis publicamente na base *HR Analytics: Job Change of Data Scientists*
(Kaggle). A empresa deseja saber quais candidatos pretendem mudar de emprego,
pois isso afeta custos de recrutamento e o planejamento dos cursos.

### Questões de pesquisa

- **Q1.** O número médio de horas de treinamento concluídas difere da carga de
  referência de 60 horas usada no planejamento das trilhas de capacitação?
- **Q2.** Mais de 20% dos candidatos (taxa de referência de rotatividade do
  setor de tecnologia) buscam uma nova oportunidade de emprego?
- **Q3.** O grau de desenvolvimento da cidade do candidato (CDI) e sua
  experiência profissional estão associados à intenção de mudança?
- **Q4.** Possuir experiência relevante na área está associado à intenção de
  mudança?
- **Q5.** O tipo de empresa em que o candidato trabalha está associado à
  intenção de mudança?
- **Q6.** O nível de escolaridade está associado à intenção de mudança?
- **Q7.** Estudantes matriculados em curso universitário integral buscam
  mudança em maior proporção do que candidatos sem matrícula?
- **Q8.** A distribuição das faixas de experiência na amostra adere à
  distribuição populacional conhecida (verificação de representatividade)?

### Variáveis analisadas

| Variável | Tipo | Descrição |
|---|---|---|
| `training_hours` | Quantitativa discreta | Horas de treinamento concluídas |
| `target` | Qualitativa nominal (binária) | Busca nova oportunidade? (1 = sim; 0 = não) |
| `city_development_index` (CDI) | Quantitativa contínua | Índice de desenvolvimento da cidade (0 a 1) |
| `relevent_experience` | Qualitativa nominal (binária) | Possui experiência relevante na área |
| `company_type` | Qualitativa nominal | Tipo de empresa (agrupada em 4 categorias) |
| `education_level` | Qualitativa ordinal | Nível de escolaridade (5 níveis) |
| `experience` | Qualitativa ordinal | Anos de experiência (agrupada em 5 faixas) |
| `enrolled_university` | Qualitativa nominal | Situação de matrícula universitária (3 níveis) |

## 2. Metodologia

### 2.1 Planejamento amostral

- **Nível de confiança:** 95% (α = 0,05); **poder-alvo dos testes:** 80%.
- **Método de amostragem:** amostragem aleatória simples **sem reposição**,
  com sementes fixadas para reprodutibilidade (piloto: 123; amostra final: 2026).
- **Amostra piloto (n = 50):** usada para estimar o desvio-padrão de
  `training_hours` (s = 55,82 h) e a fração de registros sem informação de
  empresa (38%), parâmetros necessários ao dimensionamento.

O tamanho amostral foi calculado para atender simultaneamente a três
requisitos, adotando-se o maior valor:

1. **Média** — detectar diferença mínima de Δ = 10 h em teste bilateral com
   poder de 80%: n = [(z₀,₉₇₅ + z₀,₈₀)·s/Δ]², refinado iterativamente com
   quantis da t de Student e corrigido para população finita
   (n·N/(n+N−1)) → **n = 244**;
2. **Proporção** — margem de erro E = 0,04 com 95% de confiança no cenário
   conservador p = 0,5: n = z²₀,₉₇₅·p(1−p)/E², com correção para população
   finita → **n = 583**;
3. **Testes qui-quadrado** — detectar efeito de magnitude w = 0,15
   (pequeno-médio na escala de Cohen) com poder de 80%, pela distribuição
   qui-quadrado não-central (`GofChisquarePower`), considerando os graus de
   liberdade de cada tabela planejada e inflacionando pela fração de dados
   faltantes de `company_type` estimada no piloto (485/0,62 ≈ 782), com
   correção para população finita → **n = 752**.

**Tamanho final: n = max(244; 583; 752) = 752** (3,9% da população).

### 2.2 Coleta de dados

- **Procedimento:** sorteio aleatório de 752 registros da base populacional
  (`aug_train.csv`), sem reposição;
- **Instrumento:** base secundária pública (Kaggle), coletada pela própria
  empresa no processo de inscrição nos treinamentos;
- **Variáveis derivadas:** `faixa_experiencia` (<1, 1-5, 6-10, 11-20, >20 anos),
  `tipo_empresa` (Empresa privada; Startup = Funded + Early Stage; Setor
  público; ONG/Outros = NGO + Other) e `sem_info_empresa` (flag de ausência);
- **Limitações do processo:** ver Seção 4 (vieses).

### 2.3 Procedimentos de inferência

Estimação pontual (x̄, p̂), estimação intervalar (IC 95% t-Student para μ;
Wald e Clopper-Pearson para p) e oito testes de hipóteses com α = 0,05:
teste t para uma média (Q1), teste z para uma proporção (Q2), teste t de Welch
para duas médias independentes (Q3a), testes qui-quadrado de independência
(Q3b, Q4, Q5, Q6) — com verificação da "regra dos 5" e cálculo do V de Cramér
como medida de força de associação —, teste z para duas proporções com
proporção combinada (*pooled*) e IC 95% para a diferença p₁ − p₂ (Q7), e teste
qui-quadrado de aderência com proporções de referência fixadas a priori pelo
censo populacional (Q8). Análises em Python 3 (pandas, numpy, scipy,
statsmodels, matplotlib, seaborn).

## 3. Resultados

### 3.1 Análise descritiva

A amostra final contém 752 candidatos. As horas de treinamento apresentam
média de 63,73 h, mediana de 48 h e desvio-padrão de 57,30 h, com forte
assimetria à direita (coeficiente 1,89) — Figuras 1 e 5. Na amostra, 195
candidatos (25,9%) buscam mudança de emprego (Figura 2); 74,1% possuem
experiência relevante na área; 62,8% têm graduação e 22,2% mestrado. Entre os
que informaram o tipo de empresa (68,8% da amostra), predominam empresas
privadas (54,1% do total). O CDI médio é visivelmente menor entre os que
buscam mudança (0,761 vs 0,856 — Figura 3), e a taxa de busca por mudança cai
monotonicamente com a experiência (Figura 8) e é mais alta entre quem não tem
informação de empresa (36,2% vs 21,3%).

### 3.2 Estimação pontual

- Média amostral de horas de treinamento: **x̄ = 63,73 h** (estimador não
  viesado de μ), com s = 57,30 h;
- Proporção amostral que busca mudança: **p̂ = 0,2593** (195/752).

### 3.3 Estimação intervalar (95%)

- **IC95%(μ) = (59,63; 67,83) horas** — t-Student, 751 gl, margem de 4,10 h;
- **IC95%(p):** Wald (0,2280; 0,2906); Clopper-Pearson exato (0,2283; 0,2922).
  A condição n·p̂·(1−p̂) = 144,4 ≥ 5 valida a aproximação normal.

Interpretação: com 95% de confiança, o procedimento gera intervalos que contêm
a média populacional de horas de treinamento; o intervalo obtido foi de 59,6 a
67,8 horas. Analogamente, o intervalo para a proporção que busca mudança foi
de 22,8% a 29,1%.

### 3.4 Testes de hipóteses (α = 0,05)

| # | Hipóteses | Estatística | p-valor | Decisão |
|---|---|---|---|---|
| Q1 | H₀: μ = 60 vs H_A: μ ≠ 60 | t = 1,785 (751 gl) | 0,0747 | Não rejeita H₀ |
| Q2 | H₀: p = 0,20 vs H_A: p > 0,20 | z = 4,066 | 0,000024 | **Rejeita H₀** |
| Q3a | H₀: μ_CDI busca = μ_CDI não busca | t = −8,239 (Welch, 262 gl) | 8,3×10⁻¹⁵ | **Rejeita H₀** |
| Q3b | H₀: experiência ⊥ mudança | χ² = 22,30 (4 gl); V = 0,173 | 0,00017 | **Rejeita H₀** |
| Q4 | H₀: exp. relevante ⊥ mudança | χ² = 11,60 (1 gl); V = 0,124 | 0,00066 | **Rejeita H₀** |
| Q5 | H₀: tipo de empresa ⊥ mudança | χ² = 4,66 (3 gl); V = 0,095 | 0,199 | Não rejeita H₀ |
| Q5c | H₀: ausência de info de empresa ⊥ mudança | χ² = 17,89 (1 gl); V = 0,154 | 0,000023 | **Rejeita H₀** |
| Q6 | H₀: escolaridade ⊥ mudança | χ² = 11,68 (4 gl); V = 0,126 | 0,0199 | **Rejeita H₀** |
| Q7 | H₀: p_integral = p_sem matrícula vs H_A: > | z = 5,678 | 6,8×10⁻⁹ | **Rejeita H₀** |
| Q8 | H₀: amostra adere à distribuição populacional de experiência | χ² = 3,56 (4 gl) | 0,469 | Não rejeita H₀ |

**Q1.** Não há evidência de que a carga média difira de 60 h (p = 0,075). O
estudo foi dimensionado para detectar diferenças de pelo menos 10 h; diferenças
menores (a estimativa pontual foi de +3,7 h) podem existir sem serem detectadas
— ausência de evidência não é evidência de ausência.

**Q2.** Há evidência forte de que mais de 20% dos candidatos buscam mudança
(p̂ = 25,9%, p-valor < 0,001).

**Q3.** O CDI médio de quem busca mudança é significativamente menor
(0,761 vs 0,856; p ≈ 10⁻¹⁵): candidatos de cidades menos desenvolvidas são
mais propensos a buscar novas oportunidades. A experiência também está
associada (p < 0,001): a taxa de busca cai de 56% (<1 ano, na amostra) para
16% (>20 anos), sugerindo que a mobilidade se concentra no início de carreira.

**Q4.** Candidatos **sem** experiência relevante buscam mudança em maior
proporção (35,4% vs 22,6%; p < 0,001).

**Q5.** Entre os candidatos que **informaram** o tipo de empresa, não há
evidência de associação entre o tipo (privada, startup, setor público,
ONG/outros) e a intenção de mudança (p = 0,199; V = 0,095). Contudo, a
**ausência** da informação de empresa está fortemente associada ao desfecho
(36,2% vs 21,3% de busca; p < 0,001): trata-se de ausência informativa — o
não preenchimento provavelmente indica candidatos sem vínculo empregatício
formal, grupo naturalmente mais móvel.

**Q6.** Há associação entre escolaridade e intenção de mudança (p = 0,020),
com maior taxa entre graduados (29,9% na amostra) e menores taxas nos extremos
(ensino básico e doutorado) — padrão compatível com o perfil de quem está em
transição de carreira via requalificação.

**Q7.** Estudantes de curso integral buscam mudança em proporção muito maior
que candidatos sem matrícula: p̂₁ = 44,3% (n₁ = 140) contra p̂₂ = 20,9%
(n₂ = 556), z = 5,68, p < 0,001. O IC 95% para a diferença, (0,145; 0,323),
indica que a vantagem é de pelo menos 14,5 pontos percentuais — o maior efeito
observado entre as variáveis testadas, coerente com a leitura de que a
matrícula integral sinaliza investimento ativo em transição de carreira.

**Q8.** O teste de aderência não rejeita H₀ (χ² = 3,56; 4 gl; p = 0,469): a
distribuição das faixas de experiência na amostra é compatível com as
proporções populacionais conhecidas, corroborando empiricamente a
representatividade da amostragem aleatória simples utilizada.

### 3.5 Verificação das condições de aplicação

- **Independência:** AAS sem reposição com n/N = 3,9% (< 10%) — observações
  aproximadamente independentes;
- **Normalidade de `training_hours`:** rejeitada pelo teste de Lilliefors
  (p = 0,001) e visível nas Figuras 1 e 5; contudo, com n = 752 o **Teorema
  Central do Limite** assegura a normalidade aproximada da distribuição
  amostral de x̄, validando o IC e o teste t;
- **Proporções:** n·p₀ = 150 e n·(1−p₀) = 602, ambos ≥ 5;
- **Qui-quadrado (independência e aderência, "regra dos 5"):** nenhuma
  frequência esperada < 1 e no máximo 20% das células com esperada < 5 em
  todos os testes — garantido pelo agrupamento prévio de categorias raras de
  `company_type` (Startup = Funded + Early Stage; ONG/Outros = NGO + Other);
  no teste de aderência, menor esperada = 20,4;
- **Duas proporções (Q7):** grupos desbalanceados (n₁ = 140 vs n₂ = 556), mas
  as condições nᵢ·p̂·(1−p̂) ≥ 5 são amplamente atendidas (26,6 e 105,8).

## 4. Discussão

Os resultados desenham um perfil coerente: a intenção de mudança concentra-se
em candidatos **em início de carreira** (Q3b), **sem experiência relevante na
área** (Q4), **de cidades menos desenvolvidas** (Q3a), **sem vínculo
empregatício informado** (Q5c) e **matriculados em curso universitário
integral** (Q7) — exatamente o perfil de quem usa treinamentos como porta de
entrada para a área de dados. Para a empresa, isso sugere que o
risco de perda (ou a oportunidade de recrutamento) não está em um "tipo de
empresa concorrente", mas na situação ocupacional e no contexto regional do
candidato.

A comparação com os valores populacionais — possível porque a população é
conhecida — valida os procedimentos: μ = 65,37 h ∈ IC95% (59,63; 67,83);
p = 0,2493 ∈ IC95% (0,2280; 0,2906); e os padrões por grupo da amostra
reproduzem os populacionais (por exemplo, taxa de busca de 38,8% entre quem
não informa empresa vs 18,4% entre quem informa, na população).

### Vieses e limitações

1. **Viés de seleção:** a população é autosselecionada — pessoas inscritas em
   treinamentos de requalificação, naturalmente mais móveis. As conclusões
   valem para candidatos desses treinamentos, **não** para profissionais de
   dados ou trabalhadores em geral.
2. **Concentração geográfica e natureza do CDI:** poucas cidades concentram
   grande parte dos registros e o CDI é um *proxy socioeconômico regional*,
   não um atributo individual. A associação da Q3a é efeito de contexto.
3. **Ausência informativa (MNAR):** `company_type`/`company_size` faltam
   juntas em cerca de 1/3 dos registros e a ausência está associada ao
   desfecho (Q5c). Por isso ela foi tratada como categoria de análise própria
   — descartá-la silenciosamente enviesaria a Q5 e ocultaria o achado mais
   relevante.
4. **Variável `gender`:** excluída dos testes por ter 23,5% de dados faltantes
   e forte desbalanceamento (Male ≈ 69%, Female ≈ 6,5%), o que fragilizaria as
   conclusões e criaria risco de leitura discriminatória em contexto de RH
   (inclusive por proxy, já que a ausência de informação de empresa se
   distribui de forma desigual entre gêneros).
5. **Viés de rótulo:** a definição operacional de "busca mudança" não é
   documentada pela fonte (auto-relato ou comportamento observado?), uma
   ameaça à validade de constructo que não podemos verificar.
6. **Categorias agrupadas:** os agrupamentos de `company_type` e das faixas de
   experiência foram necessários para a validade do qui-quadrado, mas implicam
   perda de granularidade.

## 5. Conclusões

Com amostra aleatória simples de n = 752 (dimensionada para média, proporção e
qui-quadrado, com correção para população finita), estimou-se que os candidatos
concluem em média 63,7 h de treinamento (IC95%: 59,6 a 67,8) e que 25,9%
buscam nova oportunidade (IC95%: 22,8% a 29,1%). Ao nível de 5%: **(Q1)** não
há evidência de que a carga média difira de 60 h; **(Q2)** mais de 20% dos
candidatos buscam mudança; **(Q3)** menor desenvolvimento da cidade e menor
experiência estão associados à busca por mudança; **(Q4)** a ausência de
experiência relevante também; **(Q5)** o tipo de empresa, entre quem o
informa, **não** apresenta associação detectável — o sinal está na ausência da
informação, que é ela própria informativa; **(Q6)** a escolaridade está
associada, com pico entre graduados; **(Q7)** estudantes de curso integral
buscam mudança em proporção muito superior à de candidatos sem matrícula
(diferença de ao menos 14,5 p.p., o maior efeito do estudo); e **(Q8)** o
teste de aderência confirma que a amostra reproduz a distribuição populacional
de experiência, validando o desenho amostral.

**Trabalhos futuros:** modelar o desfecho com regressão logística controlando
simultaneamente os fatores (evitando dupla contagem entre variáveis
redundantes como cidade × CDI); avaliar equidade entre gêneros com dados mais
completos; e investigar a definição operacional do rótulo junto à fonte.

## Referências

- Material didático da disciplina Análise Inferencial (unidades 2 a 8),
  StatLab. Disponível em: https://statlab-oficial.github.io/AnaliseInferencial/
- KAGGLE. *HR Analytics: Data Scientist Job Change Factors*. Disponível em:
  https://www.kaggle.com/datasets/obaidhere/hr-analytics-data-scientist-job-change-factors
- CASELLA, G.; BERGER, R. L. *Statistical Inference*. 2. ed. Duxbury Press, 2002.
- MAGALHÃES, M. N. *Probabilidade e variáveis aleatórias*. 2. ed. EDUSP, 2006.
- COHEN, J. *Statistical Power Analysis for the Behavioral Sciences*. 2. ed.
  Lawrence Erlbaum, 1988.
