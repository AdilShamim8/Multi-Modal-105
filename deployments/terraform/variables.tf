# Tunables for the generation-platform cluster.
# GPU sizing note: one L4 (~$0.70/h spot-equivalent) sustains roughly 2.5
# 512px/30-step T2I jobs/s — see docs/operations/capacity-planning.md for
# the full fleet math before changing gpu_max_nodes.

variable "project_id" {
  type        = string
  description = "GCP project hosting the platform."
}

variable "region" {
  type        = string
  default     = "us-east1"
  description = "Primary region (regional cluster = 3 zones of HA)."
}

variable "environment" {
  type        = string
  default     = "prod"
  description "Environment label; separate state per environment."
}

variable "k8s_version" {
  type        = string
  default     = "1.31"
  description = "GKE minor version (REGULAR channel keeps patch level)."
}

variable "cpu_machine_type" {
  type        = string
  default     = "n2-standard-8"
  description = "API tier machines: FastAPI + uvicorn workers are cheap."
}

variable "cpu_nodes_per_zone" {
  type        = number
  default     = 2
  description "Baseline CPU nodes per zone (HPA scales pods, not nodes)."
}

variable "gpu_machine_type" {
  type        = string
  default     = "g2-standard-8"
  description "L4 host: 1 GPU + 8 vCPU + 32GB — fits one worker pod."
}

variable "gpu_type" {
  type        = string
  default     = "nvidia-l4"
  description "Accelerator type (L4 = cost/efficiency sweet spot for DiT-S/B)."
}

variable "gpu_min_nodes" {
  type        = number
  default     = 1
  description "Warm floor per zone — bounds cold-start p95 latency."
}

variable "gpu_max_nodes" {
  type        = number
  default     = 20
  description "Hard fleet cap (3 zones): the platform's cost circuit-breaker."
}
