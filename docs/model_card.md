# Model Card — Telco Customer Churn Prediction

## Model Details

| Campo | Valor |
|-------|-------|
| **Nome** | ChurnMLP |
| **Versão** | 1.0.0 |
| **Tipo** | Multi-Layer Perceptron (PyTorch) |
| **Arquitetura** | Linear(input→64) → BN → ReLU → Dropout(0.3) → Linear(64→32) → BN → ReLU → Dropout(0.3) → Linear(32→16) → BN → ReLU → Dropout(0.3) → Linear(16→1) |
| **Loss** | BCEWithLogitsLoss com pos_weight (balanceamento de classe) |
| **Otimizador** | Adam (lr=1e-3, weight_decay=1e-4) + ReduceLROnPlateau |
| **Treinamento** | Early stopping (patience=15, baseado em val_loss) |
| **Seed** | 42 (reprodutibilidade total) |
| **Desenvolvido por** | FIAP 10MLET — Tech Challenge Fase 1 |

## Uso Pretendido

**Caso de uso primário:** Classificar clientes de uma operadora de telecomunicações quanto ao risco de cancelamento (churn) nos próximos meses, permitindo que a equipe de retenção priorize ações preventivas.

**Usuários pretendidos:** Equipe de CRM/retenção, cientistas de dados da operadora.

**Fora do escopo:**
- Predição de churn em outros setores (banco, SaaS, etc.) sem re-treino.
- Tomada de decisão automatizada sem revisão humana para casos de alto impacto.
- Clientes com menos de 1 mês de tenure (comportamento instável).

## Dataset de Treinamento

| Atributo | Valor |
|----------|-------|
| **Fonte** | IBM Telco Customer Churn (UCI) |
| **Tamanho** | 7.043 registros |
| **Features usadas** | 19 (3 numéricas + 5 binárias + 11 categóricas) |
| **Target** | `Churn Value` (0 = não cancelou, 1 = cancelou) |
| **Balanceamento** | ~26% positivos (churn), ~74% negativos |
| **Split** | 68% treino / 12% validação / 20% teste (estratificado) |

**Features removidas por data leakage:**
- `Churn Score`, `CLTV`, `Churn Reason` — geradas após o evento de churn.
- `CustomerID`, `Lat Long`, coordenadas geográficas — identificadores sem poder preditivo.

## Métricas de Performance (conjunto de teste)

| Modelo | AUC-ROC | F1 | PR-AUC | Accuracy |
|--------|---------|----|--------|----------|
| DummyClassifier | ~0.50 | ~0.27 | ~0.27 | ~0.73 |
| LogisticRegression | ~0.84 | ~0.60 | ~0.68 | ~0.80 |
| RandomForest | ~0.83 | ~0.60 | ~0.66 | ~0.80 |
| RF Tuned (RandomizedSearchCV) | ~0.85 | ~0.62 | ~0.69 | ~0.81 |
| GradientBoosting | ~0.85 | ~0.62 | ~0.70 | ~0.81 |
| **ChurnMLP** | **~0.86** | **~0.63** | **~0.72** | **~0.81** |

*Execute `make train` para métricas exatas no seu ambiente.*

**Métrica primária:** AUC-ROC — mede discriminação independente do threshold, adequada para datasets desbalanceados.

**Trade-off de custo:**
- Falso Negativo (churn não detectado): alto custo — cliente perdido sem intervenção.
- Falso Positivo (churn previsto erroneamente): baixo custo — ação de retenção desnecessária.
- Recomenda-se threshold abaixo de 0.5 para maximizar recall em contextos onde o custo de FN é dominante.

## Limitações

1. **Distribuição temporal:** O dataset é um snapshot estático. Mudanças no comportamento dos clientes ao longo do tempo (concept drift) degradam o modelo sem re-treino.
2. **Viés geográfico:** Dados de uma única operadora em região específica. Generalização para outros mercados requer validação.
3. **Features ausentes:** Histórico de chamadas de suporte, NPS, e interações em canais digitais não estão disponíveis e poderiam melhorar a performance.
4. **Clientes novos:** Clientes com `Tenure Months` < 3 têm comportamento mais variável; o modelo pode ser menos confiável nesse segmento.
5. **Balanceamento:** A classe positiva (churn) representa ~26%. O modelo pode sub-detectar padrões de churn raros.

## Considerações de Viés e Fairness

- **Gênero:** A feature `Gender` está incluída; recomenda-se monitorar se a taxa de falsos negativos difere entre grupos de gênero.
- **Senior Citizen:** Clientes idosos podem ter padrões de uso distintos. Validar métricas separadamente para este segmento.
- **Plano de ação de Fairness:** Executar análise com Fairlearn após deploy inicial e verificar demographic parity por contrato (`Contract`) e tipo de serviço.

## Implantação

- **Modo:** Real-time via FastAPI (`POST /predict`) ou batch via (`POST /predict-batch`).
- **Threshold padrão:** 0.5 (configurável via variável de ambiente `PREDICTION_THRESHOLD`).
- **Latência esperada:** < 50ms por predição (CPU).
- **Dependências:** ver `pyproject.toml`.

## Monitoramento

- Monitorar distribuição de `churn_probability` em produção semanalmente.
- Alerta se F1 cair mais de 5 pontos percentuais vs. baseline.
- Retreinar mensalmente ou quando drift for detectado (PSI > 0.2 ou KS p-value < 0.05).
- Módulo de drift: `src/monitoring/drift_detection.py` (KS test + PSI por feature numérica).

## Manutenção

- **Re-treino recomendado:** Mensal ou quando AUC-ROC no tráfego real cair > 3 pontos percentuais.
- **Responsável:** Equipe de ML Engineering — FIAP 10MLET Tech Challenge Fase 1.
- **Feedback loop:** Coletar ground-truth de churn 30-60 dias após predição para calcular métricas reais.
