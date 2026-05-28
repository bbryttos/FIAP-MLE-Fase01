# Plano de Monitoramento — Telco Churn API

## Objetivos

Detectar degradação de performance, data drift e problemas operacionais antes que impactem decisões de negócio.

---

## 1. Métricas Operacionais (infraestrutura)

| Métrica | Threshold de Alerta | Ação |
|---------|--------------------|----|
| Latência p95 (`/predict`) | > 200ms | Investigar gargalo; escalar instância se necessário |
| Taxa de erro HTTP 5xx | > 1% em 5 min | Verificar logs da API; rollback se > 5% |
| Taxa de erro HTTP 4xx | > 10% em 5 min | Revisar schema da request; possível mudança upstream |
| Throughput (req/min) | < 10% do baseline | Verificar se client está enviando dados |

**Coleta:** Middleware de latência já integrado na API (`X-Latency-Ms` header + logs estruturados).

---

## 2. Métricas de Modelo (performance)

Calculadas quando o ground-truth (churn real) estiver disponível (30-60 dias após predição).

| Métrica | Baseline (treino) | Alerta | Ação |
|---------|------------------|--------|------|
| AUC-ROC | ~0.86 | < 0.80 | Re-treinar com dados recentes |
| F1-Score | ~0.63 | < 0.55 | Revisar threshold ou re-treinar |
| PR-AUC | ~0.72 | < 0.62 | Verificar distribuição do target |
| Taxa de churn real vs. previsto | ~26% | Desvio > 5pp | Verificar concept drift |

---

## 3. Data Drift (features)

Monitorar distribuição das features de entrada em produção vs. distribuição de treino.

| Feature | Método | Threshold |
|---------|--------|-----------|
| `Tenure Months` | KS test | p-value < 0.05 |
| `Monthly Charges` | KS test | p-value < 0.05 |
| `Contract` | Chi-quadrado | p-value < 0.05 |
| `Internet Service` | Chi-quadrado | p-value < 0.05 |
| `Churn probability (output)` | KS test vs. baseline | p-value < 0.05 |

**Ferramenta recomendada:** [Evidently AI](https://www.evidentlyai.com/) — já é dependência nos exemplos da Aula 5.

**Frequência:** Semanal (ou com 500+ predições acumuladas, o que vier primeiro).

---

## 4. Monitoramento de Fairness

Executar mensalmente ou após cada re-treino:

- Calcular **demographic parity** e **equalized odds** por:
  - `Gender` (Male vs. Female)
  - `Senior Citizen` (Yes vs. No)
  - `Contract` (Month-to-month vs. contratos longos)

**Threshold:** Diferença de taxa de predição positiva entre grupos > 10pp requer revisão.

---

## 5. Logs Estruturados

Cada requisição ao `/predict` deve ser registrada com:

```json
{
  "timestamp": "2026-05-20T14:32:00Z",
  "request_id": "uuid",
  "churn_probability": 0.73,
  "churn_prediction": true,
  "threshold": 0.5,
  "latency_ms": 12.4,
  "model_version": "0.1.0"
}
```

Armazenar em sistema de log centralizado (ex.: CloudWatch, Datadog, ELK).

---

## 6. Playbook de Resposta a Incidentes

### AUC-ROC abaixo do threshold

1. Verificar se houve mudança na distribuição das features (data drift).
2. Coletar novos dados de treino dos últimos 60-90 dias.
3. Re-treinar com dados recentes; comparar métricas no conjunto de holdout.
4. Registrar novo experimento no MLflow antes de promover.
5. Deploy gradual (canary): 10% → 50% → 100% do tráfego.

### Alta taxa de erros 5xx

1. Verificar logs da API por stack traces.
2. Verificar se artefatos de modelo estão acessíveis (path correto, permissões).
3. Rollback para versão anterior via MLflow Model Registry se necessário.

### Data drift detectado

1. Identificar qual feature driftou (gráfico de distribuição vs. referência).
2. Consultar time de dados sobre possível mudança upstream (ex.: novo produto, campanha).
3. Decidir entre: (a) re-treinar com dados recentes, (b) ajustar threshold, (c) aguardar normalização.

---

## 7. Frequência de Re-treino

| Gatilho | Ação |
|---------|------|
| AUC-ROC real < 0.80 | Re-treino imediato |
| Data drift em ≥ 3 features | Re-treino em 1 semana |
| Ciclo calendário | Re-treino mensal preventivo |
| Mudança significativa no produto/preços | Re-treino imediato |
