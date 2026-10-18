# Outputs consumed by CI (kubeconfig wiring) and capacity dashboards.

output "cluster_name" {
  value       = google_container_cluster.multimodal.name
  description "Name of the regional GKE cluster."
}

output "cluster_endpoint" {
  value       = google_container_cluster.multimodal.endpoint
  description "API server endpoint (public; auth via RBAC + certs)."
  sensitive   = true
}

output "cluster_ca_certificate" {
  value       = google_container_cluster.multimodal.master_auth[0].cluster_ca_certificate
  description "CA cert for kubeconfig generation in CI."
  sensitive   = true
}

output "gpu_node_pool" {
  value = {
    name     = google_container_node_pool.gpu.name
    min_size = var.gpu_min_nodes
    max_size = var.gpu_max_nodes
  }
  description "GPU pool bounds — mirror these in the KEDA maxReplicaCount."
}

output "nat_router" {
  value       = google_compute_router_nat.multimodal.name
  description "NAT router serving worker egress to S3/model registry."
}
