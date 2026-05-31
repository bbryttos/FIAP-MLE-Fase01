# Deploy AWS com Terraform

Este documento centraliza a documentacao de infraestrutura em nuvem para o projeto.

Escopo atual:

- ECS Fargate para a API
- API Gateway HTTP para exposicao publica da API
- ALB como backend da API dentro da stack
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

## Fluxo de provisionamento

### 1) Preparar variaveis

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
```

Preencha ao menos:

- `aws_profile` (se estiver usando perfil)
- `container_image`

### 2) Criar infraestrutura

```bash
terraform init
terraform fmt -recursive
terraform validate
terraform plan -out tfplan
terraform apply tfplan
```

Explicacao dos comandos:

| Comando | O que faz | Quando usar |
|---|---|---|
| `terraform init` | Inicializa o diretorio Terraform, baixa providers/modulos e prepara o ambiente local. | Na primeira execucao do projeto ou quando houver mudanca de providers/modulos/backend. |
| `terraform fmt -recursive` | Formata os arquivos `.tf` no padrao oficial do Terraform, inclusive em subpastas. | Sempre antes de validar ou versionar alteracoes de IaC. |
| `terraform validate` | Valida sintaxe e consistencia da configuracao, sem criar recursos na AWS. | Depois de alterar arquivos Terraform e antes do `plan`. |
| `terraform plan -out tfplan` | Gera o plano de mudancas e salva no arquivo `tfplan`. | Para revisar exatamente o que sera criado/alterado/removido. |
| `terraform apply tfplan` | Aplica exatamente o plano salvo em `tfplan`, sem recalcular mudancas. | Depois de revisar o plano e aprovar a execucao. |

Outputs principais:

- `api_gateway_url`
- `api_docs_url`
- `api_health_url`
- `api_ready_url`
- `alb_dns_name`
- `ecr_repository_url`
- `ecs_cluster_name`
- `ecs_service_name`

## Publicar imagem no ECR

Com a infraestrutura criada, publique a imagem no repositorio ECR:

```bash
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=$(terraform output -raw ecr_repository_url | cut -d'.' -f4)
ECR_REPO=$(terraform output -raw ecr_repository_url)

aws ecr get-login-password --region "${AWS_REGION}" | \
docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
cd ../..
```

### Build e push por sistema operacional

#### macOS Apple Silicon (M1/M2/M3 - arm64)

Use build direcionado para `linux/amd64` (arquitetura esperada no ECS/Fargate neste projeto):

```bash
docker buildx create --use --name churn-amd64-builder 2>/dev/null || true
docker buildx build \
  --platform linux/amd64 \
  -f Dockerfile \
  -t "${ECR_REPO}:latest" \
  --push .
```

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
- `GET /health`: disponibilidade basica.
- `GET /ready`: readiness da API com artefatos carregados.
- Verificar logs no CloudWatch do grupo `/ecs/<project>-<env>-api`.

## Evolucoes recomendadas

- HTTPS com ACM + listener 443 no ALB.
- Ambientes `staging` e `prod` separados.
- CI/CD com OIDC do GitHub Actions (sem access key estatica).
- Auto Scaling do ECS Service por CPU/memoria/throughput.

