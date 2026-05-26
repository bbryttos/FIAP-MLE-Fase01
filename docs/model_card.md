# Model Card — Churn Prediction MLP

## Descricao

Rede neural MLP (Multi-Layer Perceptron) treinada com PyTorch para prever a probabilidade de churn de clientes de uma operadora de telecomunicacoes. O modelo e parte de um pipeline end-to-end que inclui engenharia de features, treinamento, comparacao com baselines e servico via API REST.

**Caso de uso primario:** ranquear clientes por risco de cancelamento para priorizacao de acoes preventivas de retencao pela equipe de Customer Success.

---

## Dados de Treinamento

| Atributo         | Valor                                                    |
|------------------|----------------------------------------------------------|
| Fonte            | IBM Telco Customer Churn Dataset                         |
| Registros totais | 7.043 clientes                                           |
| Split            | 70% treino / 10% validacao / 20% teste (estratificado)  |
| Taxa de churn    | 26,5% (desbalanceamento moderado)                        |
| Features raw     | 19 variaveis (demograficas, contratuais, de uso, cobranca) |
| Features apos eng.| 33 variaveis (19 raw + 14 derivadas)                   |
| Input dim (MLP)  | 59 (apos one-hot encoding das categoricas)               |
| Target           | `Churn` (binario: 0 = nao cancela, 1 = cancela)         |

### Features Derivadas (Engenharia de Features)

| Feature                  | Descricao                                      |
|--------------------------|------------------------------------------------|
| `charges_per_tenure`     | Custo medio por mes de contrato                |
| `num_services`           | Quantidade de servicos contratados             |
| `is_new_customer`        | Indicador: tenure <= 12 meses                  |
| `is_long_term_customer`  | Indicador: tenure >= 48 meses                  |
| `high_monthly_charges`   | Indicador: MonthlyCharges > percentil 75       |
| `service_bundle_score`   | Score ponderado de bundle de servicos          |
| `has_tech_support`       | Flag para suporte tecnico ativo                |
| `has_security`           | Flag para servico de seguranca online          |
| `has_backup`             | Flag para backup online                        |
| `has_protection`         | Flag para protecao de dispositivo              |
| `contract_risk_score`    | Pontuacao de risco baseada no tipo de contrato |
| `payment_risk`           | Indicador de metodo de pagamento manual        |
| `paperless_and_auto`     | Flag: fatura digital + pagamento automatico    |
| `tenure_x_monthly`       | Interacao entre tenure e custo mensal          |

---

## Arquitetura do Modelo

```
Input (59)
   │
   ├─► Linear(59→128) → BatchNorm(128) → ReLU → Dropout(0.3)
   │
   ├─► Linear(128→64) → BatchNorm(64)  → ReLU → Dropout(0.3)
   │
   ├─► Linear(64→32)  → BatchNorm(32)  → ReLU → Dropout(0.3)
   │
   └─► Linear(32→1)   [logit]
           │
        Sigmoid → churn_probability ∈ [0, 1]
```

| Componente       | Escolha             | Justificativa                                          |
|------------------|---------------------|--------------------------------------------------------|
| Camadas ocultas  | [128, 64, 32]       | Compressao gradual; evita bottleneck abrupto           |
| Ativacao         | ReLU                | Evita vanishing gradient em redes moderadas            |
| Regularizacao    | Dropout(0.3)        | Reduz overfitting em dataset pequeno (~5k treino)      |
| Normalizacao     | BatchNorm           | Estabiliza gradientes, acelera convergencia            |
| Loss             | BCEWithLogitsLoss   | Numericamente estavel para classificacao binaria       |
| Otimizador       | Adam (lr=1e-3)      | Adaptativo; padrao para MLPs em dados tabulares        |
| Early stopping   | patience=10 epochs  | Interrompe ao detectar overfitting; restaura best ckpt |

---

## Metricas (conjunto de teste — 1.409 registros, 374 churners)

### Modelo Principal: MLP PyTorch

| Metrica    | Threshold 0.50 | Threshold 0.40 (max F1) |
|------------|:--------------:|:-----------------------:|
| F1-Score   | 0.6165         | **0.6386**              |
| AUC-ROC    | 0.8429         | 0.8429                  |
| PR-AUC     | 0.6497         | 0.6497                  |
| Precisao   | 0.6443         | 0.5845                  |
| Recall     | 0.5909         | **0.7059**              |
| Acuracia   | 0.8048         | 0.7856                  |

### Matriz de Confusao

| Threshold | TP  | FP  | FN  | TN  |
|-----------|-----|-----|-----|-----|
| t = 0.50  | 221 | 122 | 153 | 913 |
| t = 0.40  | 264 | 188 | 110 | 847 |
| t = 0.10  | 356 | 547 |  18 | 488 |

---

## Comparacao com Baselines

| Modelo                  | F1     | AUC-ROC | Precisao | Recall | Acuracia |
|-------------------------|--------|---------|----------|--------|----------|
| DummyClassifier         | 0.2903 | 0.5163  | 0.2891   | 0.2914 | 0.6217   |
| Logistic Regression     | 0.5935 | 0.8468  | 0.6667   | 0.5348 | 0.8055   |
| Random Forest           | 0.5401 | 0.8200  | 0.6067   | 0.4866 | 0.7800   |
| Gradient Boosting       | 0.5939 | 0.8429  | 0.6853   | 0.5241 | 0.8098   |
| **MLP PyTorch**         | **0.6165** | **0.8429** | 0.6443 | **0.5909** | 0.8048 |

**Conclusao:** O MLP supera todos os baselines em F1 e Recall, mantendo AUC-ROC equivalente ao Gradient Boosting. A vantagem e especialmente relevante no Recall, que mede a capacidade de capturar clientes que efetivamente cancelarao.

---

## Analise de Threshold e Custo de Negocio

### Premissas de custo

| Erro          | Custo estimado | Descricao                                     |
|---------------|:--------------:|-----------------------------------------------|
| Falso Positivo (FP) | $10     | Acionar retencao para cliente que nao cancelaria |
| Falso Negativo (FN) | $500    | Perder cliente que cancela sem intervencao   |

### Impacto por threshold

| Threshold | TP  | FP  | FN  | Custo Total | Economia vs t=0.50 |
|-----------|-----|-----|-----|:-----------:|:------------------:|
| 0.50      | 221 | 122 | 153 | $77.720     | —                  |
| 0.40      | 264 | 188 | 110 | $56.880     | $20.840            |
| 0.10      | 356 | 547 |  18 | **$14.470** | **$63.250**        |

**Recomendacao operacional:** Usar threshold 0.10 para maximizar recuperacao de clientes churn (Recall=0.95), aceitando maior volume de falsos positivos. O custo total cai 81% em relacao ao threshold padrao de 0.50.

**Recomendacao conservadora:** Usar threshold 0.40 como equilibrio entre F1 e custo quando a capacidade operacional de atendimento e limitada.

---

## Uso Pretendido

- **Aplicacao:** Ranquear clientes por risco de cancelamento para acoes proativas de retencao.
- **Usuarios:** Equipe de Customer Success; analistas de CRM.
- **Integracao:** API REST (`POST /predict`) que retorna `churn_probability` e `predicted_churn`.
- **Threshold padrao na API:** 0.5 (ajustavel via parametro `threshold` na requisicao).
- **Saida:** Probabilidade continua em [0,1] + classificacao binaria com threshold configuravel.

---

## Limitacoes Conhecidas

- Treinado com dados de uma unica operadora norte-americana — pode nao generalizar para outros contextos, regioes ou perfis de clientes.
- Nao inclui variaveis de satisfacao do cliente (NPS, reclamacoes, interacoes com suporte).
- Assume estacionariedade dos padroes de churn; performance degrada com mudancas no perfil de clientes (concept drift ou data drift).
- Dataset com ~7k registros e de tamanho moderado; modelos de maior complexidade podem sofrer overfitting em producao com distribuicao diferente.
- Features de cobranca (`MonthlyCharges`, `TotalCharges`) sao sensiveis a mudancas de precificacao ou politicas comerciais.
- Nao recomendado para decisoes automaticas sem revisao humana em casos com probabilidade entre 0.35 e 0.65 (zona de incerteza).

---

## Cenarios de Falha

| Cenario                            | Impacto Esperado                          | Mitigacao                               |
|------------------------------------|-------------------------------------------|-----------------------------------------|
| Novos tipos de contrato            | Features de contrato invalidas ou nulas   | Validar schema de entrada na API        |
| Mudanca no plano de cobranca       | Distribuicao de `MonthlyCharges` desviada | Monitorar PSI mensal; alertar se > 0.2  |
| Segmento de clientes novo (ex.: empresas) | Features demograficas fora do range | Rejeitar ou sinalizar como out-of-distribution |
| Dados faltantes em features criticas | Imputacao incorreta pelo pipeline     | Validar missingness < 5% por coluna     |
| API fora do ar durante pico        | Perda de scoring em batch               | Implementar retry + fallback rule-based |

---

## Vies e Fairness

- **Atributos sensiveis presentes nos dados:** `gender` (Male/Female), `SeniorCitizen` (0/1).
- O modelo foi treinado sem exclusao desses atributos; e necessario avaliar disparidade de performance por grupo antes de uso em producao.
- **Avaliacao recomendada:** Medir `false_negative_rate` e `false_positive_rate` separadamente para cada grupo usando Fairlearn (`MetricFrame`).
- Particular atencao ao `SeniorCitizen`: este grupo tem padroes de uso e churn distintos e pode ser desproporcionalmente afetado por limiares subotimos.
- Monitorar se a taxa de intervencao (acoes de retencao disparadas) e homogenea entre grupos demograficos.

---

## Monitoramento em Producao

| Metrica                      | Frequencia | Limiar de Alerta              | Acao                         |
|------------------------------|------------|-------------------------------|------------------------------|
| Distribuicao de `churn_probability` | Semanal | PSI > 0.2 vs. baseline    | Investigar data drift        |
| F1-Score (ground truth delay) | Mensal   | Queda > 5 p.p. vs. baseline   | Acionar retreinamento        |
| Taxa de missings por feature  | Diaria    | > 5% em qualquer feature critica | Alertar engenharia de dados |
| Volume de predicoes (OOD)    | Diaria    | > 10% requests com prob. > 0.9 ou < 0.1 inesperado | Revisar distribuicao |
| Latencia da API (p95)        | Continuo  | > 500ms                       | Escalar ou otimizar inferencia |

**Politica de retreinamento:** Retreinar mensalmente com dados dos ultimos 6 meses, ou imediatamente quando PSI > 0.2 ou queda de F1 for detectada.

---

## Artefatos do Modelo

| Artefato              | Caminho                        | Descricao                                      |
|-----------------------|--------------------------------|------------------------------------------------|
| Pesos do MLP          | `models/mlp_weights.pt`        | Checkpoint PyTorch com best val loss           |
| Preprocessador        | `models/preprocessor.joblib`   | Pipeline sklearn (FeatureEngineer + Scaler + OHE) |
| Melhor baseline       | `models/best_baseline.joblib`  | Gradient Boosting (comparacao)                 |
| Configuracao          | `models/model_config.json`     | `{"input_dim": 59}`                            |
| Metricas              | `models/results.json`          | Metricas de todos os modelos no test set       |

---

## Responsaveis

| Atributo           | Valor                          |
|--------------------|--------------------------------|
| Equipe             | FIAP Tech Challenge — Fase 1   |
| Autor              | Bruno Brito                    |
| Data de treinamento | 2026-05                       |
| Versao             | 1.0.0                          |
| Framework          | PyTorch 2.x / scikit-learn 1.x |
| Experimento MLflow | `churn-prediction`             |
