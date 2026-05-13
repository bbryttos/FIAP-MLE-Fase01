# Model Card — Churn Prediction MLP

## Descricao

Rede neural MLP (Multi-Layer Perceptron) treinada com PyTorch para prever a probabilidade de churn de clientes de uma operadora de telecomunicacoes.

## Dados de Treinamento

- **Fonte:** Telco Customer Churn Dataset (IBM)
- **Tamanho:** ~7.000 registros
- **Split:** 70% treino / 10% validacao / 20% teste (estratificado)
- **Variaveis:** 19 features (demograficas, contratuais, de uso e cobranca)
- **Target:** `Churn` (binario: 0=nao cancela, 1=cancela)

## Arquitetura

- **Tipo:** MLP com 3 camadas ocultas [128, 64, 32]
- **Ativacao:** ReLU
- **Regularizacao:** Dropout (0.3) + BatchNorm
- **Loss:** BCEWithLogitsLoss
- **Otimizador:** Adam (lr=1e-3)
- **Early stopping:** patience=10 epochs

## Metricas (conjunto de teste)

| Metrica    | Valor |
|------------|-------|
| F1-Score   | —     |
| AUC-ROC    | —     |
| Precisao   | —     |
| Recall     | —     |
| Acuracia   | —     |

*(preencher apos treinamento)*

## Comparacao com Baselines

| Modelo                | F1    | AUC   |
|-----------------------|-------|-------|
| DummyClassifier       | —     | —     |
| Logistic Regression   | —     | —     |
| Random Forest         | —     | —     |
| Gradient Boosting     | —     | —     |
| **MLP PyTorch**       | **—** | **—** |

## Uso Pretendido

- Identificar clientes com alto risco de cancelamento para acoes preventivas de retencao.
- Uso interno pela equipe de Customer Success.
- Threshold padrao: 0.5 (ajustavel conforme custo de falso positivo/negativo).

## Limitacoes Conhecidas

- Treinado com dados de uma unica operadora — pode nao generalizar para outros contextos.
- Nao inclui variaveis de satisfacao do cliente (NPS, reclamacoes).
- Performance pode degradar com mudancas no perfil de clientes (data drift).
- Nao recomendado para decisoes automaticas sem revisao humana em casos limiares (prob entre 0.4 e 0.6).

## Vies e Fairness

- Atributos sensiveis presentes nos dados: `gender`, `SeniorCitizen`.
- Avaliar disparidade de performance por grupo antes de uso em producao.
- Aplicar Fairlearn para medir `false_negative_rate` por grupo.

## Monitoramento

- Monitorar distribuicao de `churn_probability` em producao semanalmente.
- Alerta se F1 cair mais de 5 pontos percentuais vs. baseline.
- Retreinar mensalmente ou quando drift for detectado (PSI > 0.2).

## Responsaveis

- **Equipe:** FIAP Tech Challenge — Fase 1
- **Data de treinamento:** a preencher
- **Versao:** 1.0.0
