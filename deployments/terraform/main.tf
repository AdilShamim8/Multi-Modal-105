# ─────────────────────────────────────────────────────────────────────────────
# Regional GKE cluster for the generation platform.
# Design decisions (see docs/adr/009-terraform-gke.md):
#   * Regional (3 zones) — control-plane HA during zonal failures.
#   * Separate GPU node pool with autoscaling — cost tracks demand, matching
#     the KEDA queue-driven worker autoscaling above it.
#   * Release channel REGULAR — patch cadence without manual upgrade ops.
# Last audited: 2026-10-18
# ─────────────────────────────────────────────────────────────────────────────
terraform {
  required_version = ">= 1.7.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
  backend "gcs" {
    bucket = "multimodal-tfstate"
    prefix = "prod/gke"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_container_cluster" "multimodal" {
  name     = "multimodal-${var.environment}"
  location = var.region # regional cluster
  version  = var.k8s_version

  release_channel {
    channel = "REGULAR"
  }

  # VPC-native networking: NetworkPolicies enforce fine-grained isolation.
  networking_mode    = "VPC_NATIVE"
  network            = google_compute_network.multimodal.id
  subnetwork         = google_compute_subnetwork.multimodal.id
  datapath_provider  = "ADVANCED_DATAPATH" # eBPF for policy enforcement

  # Workload Identity — no credential secrets for S3/GCS access from pods.
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Private nodes: workers reach S3/registries via Cloud NAT only.
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.16/28"
  }

  # Default pool = system + API tier (CPU).
  remove_default_node_pool = true
  initial_node_count       = 1

  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS", "DEPLOYMENT"]
  }
}

# ── CPU pool: API replicas, Redis, observability agents ─────────────────────
resource "google_container_node_pool" "cpu" {
  name       = "cpu-pool"
  cluster    = google_container_cluster.multimodal.id
  location   = var.region
  node_count = var.cpu_nodes_per_zone

  node_config {
    machine_type = var.cpu_machine_type
    disk_size_gb = 100
    disk_type    = "pd-balanced"
    oauth_scopes = ["cloud-platform"]
    labels = {
      workload = "api"
    }
  }
}

# ── GPU pool: generation workers (KEDA scales this 2→60) ────────────────────
resource "google_container_node_pool" "gpu" {
  name     = "gpu-pool"
  cluster  = google_container_cluster.multimodal.id
  location = var.region

  autoscaling {
    min_node_count = var.gpu_min_nodes
    max_node_count = var.gpu_max_nodes
  }

  node_config {
    machine_type = var.gpu_machine_type # L4 default (see variables.tf)
    disk_size_gb = 200
    disk_type    = "pd-balanced"
    oauth_scopes = ["cloud-platform"]

    guest_accelerator {
      type  = var.gpu_type
      count = 1
    }
    labels = {
      workload                    = "generation"
      cloud.google.com/gke-accelerator = "nvidia-l4"
    }
  }
}

# ── Networking ──────────────────────────────────────────────────────────────
resource "google_compute_network" "multimodal" {
  name                    = "multimodal-net"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "multimodal" {
  name          = "multimodal-subnet"
  ip_cidr_range = "10.2.0.0/16"
  region        = var.region
  network       = google_compute_network.multimodal.id
  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.3.0.0/16"
  }
  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.4.0.0/20"
  }
}

resource "google_compute_router" "multimodal" {
  name    = "multimodal-router"
  region  = var.region
  network = google_compute_network.multimodal.id
}

resource "google_compute_router_nat" "multimodal" {
  name                               = "multimodal-nat"
  router                             = google_compute_router.multimodal.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}
