# Dicionário de Dados — HR Analytics: Job Change of Data Scientists

## Contexto

Base pública disponibilizada no Kaggle
([obaidhere/hr-analytics-data-scientist-job-change-factors](https://www.kaggle.com/datasets/obaidhere/hr-analytics-data-scientist-job-change-factors)).
Uma empresa de Big Data e Ciência de Dados oferece treinamentos e deseja
identificar, entre os candidatos que concluíram os cursos, quais realmente
pretendem trabalhar para a empresa (ou mudar de emprego), pois isso reduz
custos de recrutamento e melhora o planejamento dos treinamentos. As
informações são coletadas no ato da inscrição.

No trabalho da disciplina, o arquivo `aug_train.csv` é tratado como
**população finita (N = 19.158)**, da qual são extraídas amostras aleatórias
simples para os procedimentos de inferência.

## Arquivos

| Arquivo | Dimensões | Conteúdo |
|---|---|---|
| `archive/aug_train.csv` | 19.158 × 14 | Base rotulada (com `target`) — usada como população no trabalho |

> O conjunto original do Kaggle inclui também `aug_test.csv` (2.129 × 13, base
> de teste **sem** a coluna `target`) e `sample_submission.csv` (2.129 × 2,
> exemplo de submissão do desafio de predição). Ambos foram **removidos deste
> repositório** por não serem utilizados na análise inferencial; permanecem
> disponíveis na página original do dataset no Kaggle.

## Dicionário de variáveis (`aug_train.csv`)

| Variável | Descrição (Kaggle) | Tipo estatístico | Domínio / níveis | % ausente |
|---|---|---|---|---|
| `enrollee_id` | Identificador único de cada candidato | Identificador (não é variável de análise) | 19.158 valores únicos | 0% |
| `city` | Código da cidade onde o candidato reside | Qualitativa nominal | 123 códigos (`city_1` … `city_180`) | 0% |
| `city_development_index` | Índice de desenvolvimento da cidade (escalonado) | Quantitativa contínua | 0,448 a 0,949 | 0% |
| `gender` | Gênero do candidato | Qualitativa nominal | `Male`, `Female`, `Other` | 23,5% |
| `relevent_experience` | Se o candidato possui experiência relevante na área | Qualitativa nominal (binária) | `Has relevent experience`, `No relevent experience` | 0% |
| `enrolled_university` | Tipo de curso universitário em que está matriculado (se houver) | Qualitativa nominal | `no_enrollment`, `Part time course`, `Full time course` | 2,0% |
| `education_level` | Maior nível de escolaridade do candidato | Qualitativa **ordinal** | `Primary School` < `High School` < `Graduate` < `Masters` < `Phd` | 2,4% |
| `major_discipline` | Principal área de formação | Qualitativa nominal | `STEM`, `Business Degree`, `Arts`, `Humanities`, `No Major`, `Other` | 14,7% |
| `experience` | Anos totais de experiência na indústria | Qualitativa **ordinal** (22 níveis) | `<1`, `1`, `2`, …, `20`, `>20` | 0,3% |
| `company_size` | Número de funcionários da empresa atual do candidato | Qualitativa **ordinal** (8 faixas) | `<10`, `10/49`, `50-99`, `100-500`, `500-999`, `1000-4999`, `5000-9999`, `10000+` | 31,0% |
| `company_type` | Tipo da empresa atual do candidato | Qualitativa nominal | `Pvt Ltd`, `Funded Startup`, `Early Stage Startup`, `Public Sector`, `NGO`, `Other` | 32,0% |
| `last_new_job` | Anos entre o emprego anterior e o atual | Qualitativa **ordinal** | `never`, `1`, `2`, `3`, `4`, `>4` | 2,2% |
| `training_hours` | Horas de treinamento concluídas | Quantitativa discreta | 1 a 336 | 0% |
| `target` | Desfecho: procura mudança de emprego? | Qualitativa nominal (binária) | `1` = busca mudança; `0` = não busca (proporção populacional de 1: 24,93%) | 0% |

Observações sobre a grafia original da base: `relevent_experience` (sic, em vez
de *relevant*) e `city_development_index` são os nomes exatos das colunas.

## Dados ausentes — atenção especial

- `company_size` e `company_type` faltam **juntas** em cerca de 1/3 dos
  registros (correlação de ausência ≈ 0,84). A ausência é **informativa
  (MNAR)**: candidatos sem essas informações têm taxa de busca por mudança de
  ~38,8%, contra ~18,4% entre os que informam — provavelmente indica ausência
  de vínculo empregatício formal. No trabalho, essa ausência é tratada como
  categoria própria (flag `sem_info_empresa`), e não descartada.
- `gender` (23,5% ausente) é fortemente desbalanceada entre os respondentes
  (Male ≈ 69%, Female ≈ 6,5%, Other ≈ 1%) e não é utilizada nos testes do
  trabalho (ver seção de vieses do relatório).

## Variáveis derivadas criadas no trabalho

| Variável derivada | Construção | Uso |
|---|---|---|
| `faixa_experiencia` | `experience` agrupada em 5 faixas: `<1`, `1-5`, `6-10`, `11-20`, `>20` | Testes qui-quadrado de independência e de aderência |
| `tipo_empresa` | `company_type` agrupada em 4 grupos: Empresa privada; Startup (= Funded + Early Stage); Setor público; ONG/Outros (= NGO + Other) | Garantir a "regra dos 5" no qui-quadrado |
| `sem_info_empresa` | Flag: `company_type` ausente vs presente | Análise da ausência informativa (MNAR) |
