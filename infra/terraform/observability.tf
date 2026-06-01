# Reutiliza o mesmo cluster ECS da API para os serviços de observabilidade.
data "aws_ecs_cluster" "main" {
  cluster_name = module.ecs_service.cluster_name
}

# Policy base para permitir que tasks ECS assumam as roles de execução e task role.
data "aws_iam_policy_document" "obs_task_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

# Roles reutilizadas pelos serviços de observabilidade (execução e runtime).
resource "aws_iam_role" "obs_task_execution" {
  name               = "${local.name_prefix}-ecs-obs-exec-role"
  assume_role_policy = data.aws_iam_policy_document.obs_task_assume_role.json
}

resource "aws_iam_role_policy_attachment" "obs_task_execution_managed" {
  role       = aws_iam_role.obs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "obs_task" {
  name               = "${local.name_prefix}-ecs-obs-task-role"
  assume_role_policy = data.aws_iam_policy_document.obs_task_assume_role.json
}

# Aberturas adicionais no SG do ECS para portas dos serviços de observabilidade.
resource "aws_security_group_rule" "ecs_ingress_mlflow_from_alb" {
  type                     = "ingress"
  from_port                = 5000
  to_port                  = 5000
  protocol                 = "tcp"
  security_group_id        = module.network.ecs_security_group_id
  source_security_group_id = module.network.alb_security_group_id
  description              = "Permite ALB acessar MLflow no ECS"
}

resource "aws_security_group_rule" "ecs_ingress_prometheus_from_alb" {
  type                     = "ingress"
  from_port                = 9090
  to_port                  = 9090
  protocol                 = "tcp"
  security_group_id        = module.network.ecs_security_group_id
  source_security_group_id = module.network.alb_security_group_id
  description              = "Permite ALB acessar Prometheus no ECS"
}

resource "aws_security_group_rule" "ecs_ingress_grafana_from_alb" {
  type                     = "ingress"
  from_port                = 3000
  to_port                  = 3000
  protocol                 = "tcp"
  security_group_id        = module.network.ecs_security_group_id
  source_security_group_id = module.network.alb_security_group_id
  description              = "Permite ALB acessar Grafana no ECS"
}

# Log groups dedicados no CloudWatch para facilitar troubleshooting por serviço.
resource "aws_cloudwatch_log_group" "mlflow" {
  name              = "/ecs/${local.name_prefix}-mlflow"
  retention_in_days = var.observability_log_retention_in_days
}

resource "aws_cloudwatch_log_group" "prometheus" {
  name              = "/ecs/${local.name_prefix}-prometheus"
  retention_in_days = var.observability_log_retention_in_days
}

resource "aws_cloudwatch_log_group" "grafana" {
  name              = "/ecs/${local.name_prefix}-grafana"
  retention_in_days = var.observability_log_retention_in_days
}

# Target groups dedicados por serviço para roteamento por path no ALB.
resource "aws_lb_target_group" "mlflow" {
  name        = substr("${local.name_prefix}-mlflow-tg", 0, 32)
  port        = 5000
  protocol    = "HTTP"
  vpc_id      = module.network.vpc_id
  target_type = "ip"

  health_check {
    path                = "/mlflow"
    protocol            = "HTTP"
    matcher             = "200-399"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 20
  }
}

resource "aws_lb_target_group" "prometheus" {
  name        = substr("${local.name_prefix}-prom-tg", 0, 32)
  port        = 9090
  protocol    = "HTTP"
  vpc_id      = module.network.vpc_id
  target_type = "ip"

  health_check {
    path                = "/prometheus/-/ready"
    protocol            = "HTTP"
    matcher             = "200-399"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 20
  }
}

resource "aws_lb_target_group" "grafana" {
  name        = substr("${local.name_prefix}-grafana-tg", 0, 32)
  port        = 3000
  protocol    = "HTTP"
  vpc_id      = module.network.vpc_id
  target_type = "ip"

  health_check {
    path                = "/grafana/api/health"
    protocol            = "HTTP"
    matcher             = "200-399"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 20
  }
}

# Regras de listener que roteiam paths do ALB para cada target group.
resource "aws_lb_listener_rule" "mlflow" {
  listener_arn = module.alb.listener_arn
  priority     = 100

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.mlflow.arn
  }

  condition {
    path_pattern {
      values = ["/mlflow", "/mlflow/*"]
    }
  }
}

resource "aws_lb_listener_rule" "prometheus" {
  listener_arn = module.alb.listener_arn
  priority     = 110

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.prometheus.arn
  }

  condition {
    path_pattern {
      values = ["/prometheus", "/prometheus/*"]
    }
  }
}

resource "aws_lb_listener_rule" "grafana" {
  listener_arn = module.alb.listener_arn
  priority     = 120

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.grafana.arn
  }

  condition {
    path_pattern {
      values = ["/grafana", "/grafana/*"]
    }
  }
}

# Definições de task Fargate para MLflow, Prometheus e Grafana.
resource "aws_ecs_task_definition" "mlflow" {
  family                   = "${local.name_prefix}-mlflow"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = tostring(var.observability_task_cpu)
  memory                   = tostring(var.observability_task_memory)
  execution_role_arn       = aws_iam_role.obs_task_execution.arn
  task_role_arn            = aws_iam_role.obs_task.arn

  container_definitions = jsonencode([
    {
      name      = "mlflow"
      image     = "ghcr.io/mlflow/mlflow:latest"
      essential = true
      command = [
        "mlflow",
        "server",
        "--backend-store-uri",
        "sqlite:///mlflow.db",
        "--default-artifact-root",
        "/mlruns",
        "--host",
        "0.0.0.0",
        "--port",
        "5000",
        "--static-prefix",
        "/mlflow"
      ]
      portMappings = [
        {
          containerPort = 5000
          hostPort      = 5000
          protocol      = "tcp"
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.mlflow.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
}

resource "aws_ecs_task_definition" "prometheus" {
  family                   = "${local.name_prefix}-prometheus"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = tostring(var.observability_task_cpu)
  memory                   = tostring(var.observability_task_memory)
  execution_role_arn       = aws_iam_role.obs_task_execution.arn
  task_role_arn            = aws_iam_role.obs_task.arn

  container_definitions = jsonencode([
    {
      name      = "prometheus"
      image     = "prom/prometheus:latest"
      essential = true
      command = [
        "--web.external-url=/prometheus",
        "--web.route-prefix=/prometheus"
      ]
      portMappings = [
        {
          containerPort = 9090
          hostPort      = 9090
          protocol      = "tcp"
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.prometheus.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
}

resource "aws_ecs_task_definition" "grafana" {
  family                   = "${local.name_prefix}-grafana"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = tostring(var.observability_task_cpu)
  memory                   = tostring(var.observability_task_memory)
  execution_role_arn       = aws_iam_role.obs_task_execution.arn
  task_role_arn            = aws_iam_role.obs_task.arn

  container_definitions = jsonencode([
    {
      name      = "grafana"
      image     = "grafana/grafana:latest"
      essential = true
      environment = [
        { name = "GF_SECURITY_ADMIN_USER", value = "admin" },
        { name = "GF_SECURITY_ADMIN_PASSWORD", value = "admin123" },
        { name = "GF_USERS_ALLOW_SIGN_UP", value = "false" },
        { name = "GF_SERVER_ROOT_URL", value = "%(protocol)s://%(domain)s/grafana/" },
        { name = "GF_SERVER_SERVE_FROM_SUB_PATH", value = "true" }
      ]
      portMappings = [
        {
          containerPort = 3000
          hostPort      = 3000
          protocol      = "tcp"
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.grafana.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
}

# Serviços ECS de observabilidade (escala controlada por observability_desired_count).
resource "aws_ecs_service" "mlflow" {
  name            = "${local.name_prefix}-mlflow-service"
  cluster         = data.aws_ecs_cluster.main.arn
  task_definition = aws_ecs_task_definition.mlflow.arn
  desired_count   = var.observability_desired_count
  launch_type     = "FARGATE"

  load_balancer {
    target_group_arn = aws_lb_target_group.mlflow.arn
    container_name   = "mlflow"
    container_port   = 5000
  }

  network_configuration {
    subnets          = module.network.public_subnet_ids
    security_groups  = [module.network.ecs_security_group_id]
    assign_public_ip = true
  }

  depends_on = [
    aws_lb_listener_rule.mlflow,
    aws_iam_role_policy_attachment.obs_task_execution_managed
  ]
}

resource "aws_ecs_service" "prometheus" {
  name            = "${local.name_prefix}-prometheus-service"
  cluster         = data.aws_ecs_cluster.main.arn
  task_definition = aws_ecs_task_definition.prometheus.arn
  desired_count   = var.observability_desired_count
  launch_type     = "FARGATE"

  load_balancer {
    target_group_arn = aws_lb_target_group.prometheus.arn
    container_name   = "prometheus"
    container_port   = 9090
  }

  network_configuration {
    subnets          = module.network.public_subnet_ids
    security_groups  = [module.network.ecs_security_group_id]
    assign_public_ip = true
  }

  depends_on = [
    aws_lb_listener_rule.prometheus,
    aws_iam_role_policy_attachment.obs_task_execution_managed
  ]
}

resource "aws_ecs_service" "grafana" {
  name            = "${local.name_prefix}-grafana-service"
  cluster         = data.aws_ecs_cluster.main.arn
  task_definition = aws_ecs_task_definition.grafana.arn
  desired_count   = var.observability_desired_count
  launch_type     = "FARGATE"

  load_balancer {
    target_group_arn = aws_lb_target_group.grafana.arn
    container_name   = "grafana"
    container_port   = 3000
  }

  network_configuration {
    subnets          = module.network.public_subnet_ids
    security_groups  = [module.network.ecs_security_group_id]
    assign_public_ip = true
  }

  depends_on = [
    aws_lb_listener_rule.grafana,
    aws_iam_role_policy_attachment.obs_task_execution_managed
  ]
}
