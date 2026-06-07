# Deploy AWS com Terraform

Este documento centraliza a documentacao de infraestrutura em nuvem para o projeto.

Escopo atual:

- ECS Fargate para a API
- API Gateway HTTP para exposicao publica da API
- ALB como backend da API dentro da stack
- MLflow, Prometheus e Grafana em ECS (roteados por path no ALB/API Gateway)
- ECR como registry oficial
- VPC, subnets e security groups
- CloudWatch Logs

## Onde este documento se encaixa no projeto

- `README.md`: visao geral e acesso rapido.
- `docs/aws_terraform_deploy.md` (este arquivo): guia completo de provisionamento.
- `infra/terraform/README.md`: referencia tecnica da estrutura de codigo Terraform.
- `docs/monitoring_plan.md`: estrategia de observabilidade e alertas (camada operacional/modelo).

## Pre-requisitos

- Conta AWS ativa.
- AWS CLI configurado localmente.
- Terraform >= 1.6.
- Docker instalado para build/push da imagem.

## Credenciais AWS (sem hardcode)

O Terraform foi configurado para usar credenciais externas ao repositorio:

1. Via perfil do AWS CLI (recomendado):

```bash
aws configure --profile seu-perfil
export AWS_PROFILE=seu-perfil
```

1. Via variaveis de ambiente:

```bash
export AWS_ACCESS_KEY_ID="<sua-access-key>"
export AWS_SECRET_ACCESS_KEY="<sua-secret-key>"
export AWS_SESSION_TOKEN="<token-opcional>"
```

Nao versione credenciais, `terraform.tfvars` reais nem arquivos de state.

### Credenciais temporarias (Access Key + Secret + Session Token)

Se sua conta/lab AWS entrega credenciais temporarias, configure os tres valores no profile:

```bash
aws configure --profile seu-perfil
aws configure set aws_session_token "<seu-session-token>" --profile seu-perfil
```

Ou por variaveis de ambiente (sessao atual):

```bash
export AWS_ACCESS_KEY_ID="<sua-access-key>"
export AWS_SECRET_ACCESS_KEY="<sua-secret-key>"
export AWS_SESSION_TOKEN="<seu-session-token>"
export AWS_REGION="us-east-1"
```

Valide sempre antes de rodar Terraform:

```bash
AWS_PROFILE=seu-perfil aws sts get-caller-identity
```

Se retornar `InvalidClientTokenId`, a credencial/token esta invalida ou expirada e precisa ser renovada.

### Troca de conta AWS (local state)

Ao trocar de conta mantendo backend local, nao reutilize o state antigo.

```bash
cd infra/terraform
rm -rf .terraform .terraform.lock.hcl terraform.tfstate terraform.tfstate.backup tfplan
terraform init -reconfigure
```

Depois, ajuste `aws_profile` e o ID da conta AWS em `container_image` no `terraform.tfvars` e execute `terraform plan`.

## Fluxo de provisionamento

### Execucao rapida (copiar e colar)

Escolha o fluxo conforme o seu ambiente:

- **Fluxo A (conta com permissao IAM para criar roles)**: Terraform cria tudo.
- **Fluxo B (lab/conta com IAM restrito)**: Terraform reutiliza roles existentes via ARN.

#### Fluxo A - conta com permissao IAM

```bash
# 1) Preparar e validar credencial
cd infra/terraform
export AWS_PROFILE=seu-perfil
export AWS_REGION=us-east-1
aws sts get-caller-identity

# 2) Primeiro apply sem tasks da API (evita erro de pull antes da imagem existir)
cp -n terraform.tfvars.example terraform.tfvars
# ajuste o terraform.tfvars e deixe desired_count = 0
terraform init -reconfigure
terraform fmt -recursive
terraform validate
terraform plan -out tfplan
terraform apply tfplan

# 3) Login ECR + obter repo (ainda em infra/terraform)
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=$(terraform output -raw ecr_repository_url | cut -d'.' -f4)
ECR_REPO=$(terraform output -raw ecr_repository_url)
echo "ECR_REPO=${ECR_REPO}"
[ -z "${ECR_REPO}" ] && echo "Erro: ecr_repository_url vazio. Rode terraform apply antes." && exit 1
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# 4) Build/push da imagem da API (na raiz do projeto)
cd ../..
docker buildx create --use --name churn-amd64-builder 2>/dev/null || true
docker buildx build \
  --platform linux/amd64 \
  --provenance=false --sbom=false \
  -f Dockerfile \
  -t "${ECR_REPO}:latest" \
  --push .

# 5) Ativar tasks da API e reaplicar infra
cd infra/terraform
# ajuste desired_count = 1 no terraform.tfvars
terraform plan -out tfplan
terraform apply tfplan
```

#### Fluxo B - IAM restrito (sem iam:CreateRole)

Mesmo fluxo acima, com um passo extra no `terraform.tfvars` antes do primeiro `plan`:

```hcl
ecs_task_execution_role_arn           = "arn:aws:iam::<account-id>:role/<ecs-exec-role-existente>"
ecs_task_role_arn                     = "arn:aws:iam::<account-id>:role/<ecs-task-role-existente>"
observability_task_execution_role_arn = "arn:aws:iam::<account-id>:role/<obs-exec-role-existente>"
observability_task_role_arn           = "arn:aws:iam::<account-id>:role/<obs-task-role-existente>"
```

Se esses campos estiverem preenchidos, o Terraform nao tenta criar novas IAM Roles para ECS/API/observabilidade.

### 1) Preparar variaveis

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
```

Preencha ao menos:

- `aws_profile` (se estiver usando perfil)
- `container_image`

Importante:

- MLflow, Prometheus e Grafana usam imagens publicas e podem subir sem imagem propria no ECR.
- A API de inferencia depende da imagem indicada em `container_image` (ou `${ecr_repository_url}:latest`).
- Se a imagem da API ainda nao existir no ECR, o service pode ser criado, mas as tasks da API podem falhar ate o push da imagem.

Configuracao padrao otimizada para menor custo:

- API em Fargate com `task_cpu=256` e `task_memory=512`.
- Observabilidade com `observability_desired_count=0` (MLflow/Prometheus/Grafana criados, mas sem tasks rodando).
- Retencao de logs reduzida para 7 dias.

Quando habilitar observabilidade (`observability_desired_count=1`), recomenda-se:

- `observability_task_cpu = 512`
- `observability_task_memory = 2048`

O `mlflow` foi configurado com `--workers 1` para reduzir consumo de memoria e evitar reinicios por OOM durante o health check do ALB.

### 2) Criar infraestrutura

```bash
terraform init
terraform fmt -recursive
terraform validate
terraform plan -out tfplan
terraform apply tfplan
```

Explicacao dos comandos:


| Comando                      | O que faz                                                                             | Quando usar                                                                            |
| ---------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `terraform init`             | Inicializa o diretorio Terraform, baixa providers/modulos e prepara o ambiente local. | Na primeira execucao do projeto ou quando houver mudanca de providers/modulos/backend. |
| `terraform fmt -recursive`   | Formata os arquivos `.tf` no padrao oficial do Terraform, inclusive em subpastas.     | Sempre antes de validar ou versionar alteracoes de IaC.                                |
| `terraform validate`         | Valida sintaxe e consistencia da configuracao, sem criar recursos na AWS.             | Depois de alterar arquivos Terraform e antes do `plan`.                                |
| `terraform plan -out tfplan` | Gera o plano de mudancas e salva no arquivo `tfplan`.                                 | Para revisar exatamente o que sera criado/alterado/removido.                           |
| `terraform apply tfplan`     | Aplica exatamente o plano salvo em `tfplan`, sem recalcular mudancas.                 | Depois de revisar o plano e aprovar a execucao.                                        |


### Ordem recomendada (conta nova)

Para evitar erro de pull da imagem da API no primeiro deploy:

1. Em `terraform.tfvars`, use `desired_count = 0`.
2. Rode `terraform apply` para criar a infraestrutura base (incluindo ECR).
3. Faça o build/push da imagem da API para o ECR (secao abaixo).
4. Altere `desired_count = 1` e rode novo `terraform apply`.

### Erros comuns de autenticacao/autorizacao

- `InvalidClientTokenId`: credenciais invalidas/expiradas (renove Access Key/Secret/Session Token).
- `UnauthorizedOperation` (ex.: `ec2:DescribeAvailabilityZones`): credencial valida, mas sem permissao IAM na conta atual.

Importante: permissoes IAM base devem ser ajustadas diretamente na conta AWS (console/policy/role). Terraform so consegue gerenciar IAM se a credencial atual ja tiver privilegios para isso.

#### Caso especifico: erro no `data "aws_availability_zones" "available"`

Se o `terraform plan` parar no `main.tf` com erro de `ec2:DescribeAvailabilityZones`, isso significa que o usuario/role atual nao tem permissao de leitura de Availability Zones.

Como resolver na AWS (Console IAM):

1. Acesse **IAM > Policies > Create policy**.
2. Crie uma policy com permissao minima para leitura de AZ:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeAvailabilityZones"
      ],
      "Resource": "*"
    }
  ]
}
```

1. Anexe essa policy na role/usuario que voce usa no Terraform (ex.: role assumida pelo lab).
2. Valide novamente no terminal:

```bash
aws ec2 describe-availability-zones --region us-east-1
```

Se esse comando funcionar, o Terraform deve passar dessa etapa.

#### Workaround para ambientes com IAM restrito (sem `iam:CreateRole`)

Se a sua role nao puder criar IAM Roles, reutilize roles ja existentes preenchendo no `terraform.tfvars`:

```hcl
ecs_task_execution_role_arn           = "arn:aws:iam::<account-id>:role/<ecs-exec-role-existente>"
ecs_task_role_arn                     = "arn:aws:iam::<account-id>:role/<ecs-task-role-existente>"
observability_task_execution_role_arn = "arn:aws:iam::<account-id>:role/<obs-exec-role-existente>"
observability_task_role_arn           = "arn:aws:iam::<account-id>:role/<obs-task-role-existente>"
```

Com esses campos preenchidos, o Terraform nao tenta criar novas roles para ECS/API/observabilidade.

Outputs principais:

- `api_gateway_url`
- `api_docs_url`
- `api_health_url`
- `api_ready_url`
- `mlflow_url`
- `prometheus_url`
- `grafana_url`
- `alb_dns_name`
- `ecr_repository_url`
- `ecs_cluster_name`
- `ecs_service_name`

## Publicar imagem no ECR

Com a infraestrutura criada, publique a imagem no repositorio ECR.
Execute este bloco dentro de `infra/terraform`:

```bash
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=$(terraform output -raw ecr_repository_url | cut -d'.' -f4)
ECR_REPO=$(terraform output -raw ecr_repository_url)
echo "ECR_REPO=${ECR_REPO}"
[ -z "${ECR_REPO}" ] && echo "Erro: ecr_repository_url vazio. Rode terraform apply antes." && exit 1

aws ecr get-login-password --region "${AWS_REGION}" | \
docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
cd ../..
```

### Build e push por sistema operacional

#### macOS Apple Silicon (M1/M2/M3 - arm64)

Use build direcionado para `linux/amd64` (arquitetura esperada no ECS/Fargate neste projeto):

> Execute estes comandos na raiz do projeto (diretorio que contem o `Dockerfile`).

```bash
docker buildx create --use --name churn-amd64-builder 2>/dev/null || true
docker buildx build \
  --platform linux/amd64 \
  --provenance=false --sbom=false \
  -f Dockerfile \
  -t "${ECR_REPO}:latest" \
  --push .
```

Flags de otimizacao:

- `--provenance=false --sbom=false`: evita gerar manifests de attestation extras (push mais rapido).
- O builder `docker-container` ja mantem **cache local** entre builds na mesma maquina, entao apenas as camadas que mudaram sao re-enviadas no push seguinte.

> Atencao ao cache de registry: `--cache-to type=registry,...,mode=max` sobe uma **segunda copia** de todas as camadas para o ECR (no log aparece como `exporting cache to registry`, que pode dobrar o tempo total). So vale a pena em **CI/multiplas maquinas**, onde nao ha cache local. Numa maquina unica, evite. Se precisar em CI, prefira `mode=min` ou cache inline:
>
> ```bash
> # CI: cache inline (sem segundo upload completo)
> docker buildx build \
>   --platform linux/amd64 \
>   --provenance=false --sbom=false \
>   --cache-from "type=registry,ref=${ECR_REPO}:buildcache" \
>   --cache-to   "type=inline" \
>   -f Dockerfile -t "${ECR_REPO}:latest" --push .
> ```

> Imagem otimizada (2 cortes em `pyproject.toml`):
> 1. **torch CPU-only** em Linux (sem bibliotecas CUDA da NVIDIA) via `[tool.uv.sources]` + indice `pytorch-cpu` — reduz ~2GB.
> 2. **Libs de treino/EDA fora do runtime**: `mlflow`, `matplotlib`, `seaborn`, `imbalanced-learn`, `openpyxl` e `pandera` ficam no extra `train` (`[project.optional-dependencies].train`). A API nao as importa, e o `Dockerfile` usa `uv sync --no-dev` (sem `--extra train`), entao elas nao entram na imagem.
>
> Nenhum dos cortes afeta o ambiente local: para treinar/EDA use `uv sync --extra dev --extra train`.

#### Linux/Windows x86_64 (amd64)

Build direto costuma ser suficiente:

```bash
docker build -f Dockerfile -t "${ECR_REPO}:latest" .
docker push "${ECR_REPO}:latest"
```

Se atualizar a imagem/tag, rode novo `terraform apply` ajustando `container_image`.

### Observacao importante sobre arquitetura da imagem

Se a imagem for enviada somente como `arm64` (comum em Mac Apple Silicon), o servico pode falhar no ECS/Fargate com:

- `CannotPullContainerError`
- `image Manifest does not contain descriptor matching platform 'linux/amd64'`

Nessa situacao, normalmente o log stream aparece no CloudWatch, mas sem eventos (`storedBytes = 0`) porque o container nao chega a iniciar.

## Reset completo da infraestrutura (apagar e recriar)

Use este fluxo quando quiser limpar toda a infra e provisionar do zero.

### 1) Destruir tudo que o Terraform criou

```bash
cd infra/terraform
terraform init
terraform destroy
```

Se voce acabou de habilitar `force_delete = true` no ECR, rode antes:

```bash
terraform plan -out tfplan
terraform apply tfplan
```

Isso atualiza o estado da infraestrutura para que o `destroy` consiga remover repositorio ECR com imagens.

Opcional para execucao nao interativa:

```bash
terraform destroy -auto-approve
```

### 2) Limpar artefatos locais (opcional)

```bash
rm -rf .terraform .terraform.lock.hcl terraform.tfstate terraform.tfstate.backup tfplan
```

Observacoes:

- Nao remova `terraform.tfvars` se quiser manter seus parametros.
- Esse passo remove apenas arquivos locais; os recursos AWS ja devem ter sido apagados no `destroy`.
- O modulo ECR usa `force_delete = true`, entao o repositorio e apagado mesmo com imagens.

### 3) Provisionar novamente

```bash
terraform init
terraform fmt -recursive
terraform validate
terraform plan -out tfplan
terraform apply tfplan
```

### 4) Reenviar imagem para o ECR recriado

Use exatamente o mesmo procedimento da secao **Publicar imagem no ECR**.

Resumo:

1. Obtenha `ecr_repository_url` via `terraform output`.
2. Faça login no ECR.
3. Execute build e push conforme o seu sistema operacional.

## Validacao pos-deploy

- `terraform output -raw api_gateway_url`: endpoint principal para consumir a API.
- `terraform output -raw api_docs_url`: endpoint de testes via Swagger.
- `terraform output -raw api_health_url`: endpoint de health check.
- `terraform output -raw api_ready_url`: endpoint de readiness.
- `terraform output -raw mlflow_url`: endpoint do MLflow.
- `terraform output -raw prometheus_url`: endpoint do Prometheus.
- `terraform output -raw grafana_url`: endpoint do Grafana (usuario/senha padrao: `admin` / `admin123`).
- `GET /health`: disponibilidade basica.
- `GET /ready`: readiness da API com artefatos carregados.
- Verificar logs no CloudWatch do grupo `/ecs/<project>-<env>-api`.

## Notas sobre Free Tier

Mesmo com tuning de custo, alguns recursos desta arquitetura normalmente geram cobranca:

- ALB (Application Load Balancer)
- API Gateway HTTP
- ECS Fargate (quando `desired_count > 0`)

Para testes com menor custo, mantenha:

- `observability_desired_count = 0` (padrao)
- `task_cpu = 256` e `task_memory = 512` (padrao)
- `desired_count = 1` apenas durante validacoes

