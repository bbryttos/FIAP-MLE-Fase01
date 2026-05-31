variable "aws_region" {
  type        = string
  description = "Regiao AWS para criar os recursos."
  default     = "sa-east-1"
}

variable "aws_profile" {
  type        = string
  description = "Perfil AWS CLI. Deixe vazio para usar variaveis de ambiente AWS_*."
  default     = ""
}

variable "project_name" {
  type        = string
  description = "Nome base do projeto para prefixar recursos."
  default     = "churn-prediction"
}

variable "environment" {
  type        = string
  description = "Ambiente de deploy (ex.: dev, staging, prod)."
  default     = "dev"
}

variable "tags" {
  type        = map(string)
  description = "Tags adicionais para os recursos."
  default     = {}
}

variable "vpc_cidr" {
  type        = string
  description = "CIDR da VPC."
  default     = "10.42.0.0/16"
}

variable "public_subnet_cidrs" {
  type        = list(string)
  description = "CIDRs para as subnets publicas."
  default     = ["10.42.1.0/24", "10.42.2.0/24"]
}

variable "container_port" {
  type        = number
  description = "Porta exposta pelo container da API."
  default     = 8000
}

variable "desired_count" {
  type        = number
  description = "Quantidade desejada de tarefas no ECS Service."
  default     = 1
}

variable "task_cpu" {
  type        = number
  description = "CPU da task ECS (unidades Fargate)."
  default     = 1024
}

variable "task_memory" {
  type        = number
  description = "Memoria da task ECS em MiB."
  default     = 2048
}

variable "container_image" {
  type        = string
  description = "Imagem Docker da API (ex.: <account>.dkr.ecr.<region>.amazonaws.com/repo:tag)."
  default     = ""
}

variable "jwt_expire_minutes" {
  type        = number
  description = "TTL do token JWT em minutos."
  default     = 60
}

variable "models_dir" {
  type        = string
  description = "Diretorio dos artefatos de modelo no container."
  default     = "models"
}

variable "log_level" {
  type        = string
  description = "Nivel de log da aplicacao."
  default     = "INFO"
}

variable "rate_limit_requests" {
  type        = number
  description = "Quantidade de requests permitida por janela."
  default     = 100
}

variable "rate_limit_window" {
  type        = number
  description = "Janela do rate limit em segundos."
  default     = 60
}

variable "health_check_path" {
  type        = string
  description = "Path de health check no ALB target group."
  default     = "/ready"
}

variable "alb_ingress_cidr" {
  type        = string
  description = "CIDR autorizado a acessar o ALB."
  default     = "0.0.0.0/0"
}

