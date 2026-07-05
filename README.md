# Trabalho Aplicado — Inferência Estatística

**Tema:** Intenção de mudança de emprego entre profissionais de Ciência de Dados |
**Aluno:** Lucas Ferreira da Silva — 567414
**Disciplina:** Análise Inferencial

Aplicação dos métodos de inferência estatística (planejamento amostral,
estimação pontual e intervalar e testes de hipóteses) sobre a base pública
*HR Analytics: Job Change of Data Scientists* (Kaggle), tratada como população
finita (N = 19.158).

## Conteúdo

- **[projeto/](projeto/)** — código, dados amostrais, figuras, resultados e
  relatório. O [README do projeto](projeto/README.md) detalha as questões de
  pesquisa (Q1 a Q8), a metodologia e como executar.
- **[RH - Novas opor/archive/](RH%20-%20Novas%20opor/archive/)** — base de
  dados populacional (`aug_train.csv`).

## Execução rápida

```bash
pip install pandas numpy scipy statsmodels matplotlib seaborn streamlit

# análise completa (gera figuras, amostras e resultados)
python projeto/codigo/analise_inferencial.py

# app interativo
streamlit run projeto/app/app_streamlit.py
```

Análise reprodutível: sementes fixadas (amostra piloto: 123; amostra final: 2026).
