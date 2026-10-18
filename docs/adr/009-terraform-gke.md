# ADR-009: Terraform + regional GKE for the reference deployment

- **Status:** accepted (2026-10-18)
- **Context:** the platform's manifests target Kubernetes, but a
    reproducible cluster underneath is what makes the topology claims
    (zones, GPU pools, private nodes) verifiable instead of aspirational.
- **Decision:** Terraform (google ~> 6.0) describing a **regional** GKE
  cluster: REGULAR release channel, VPC-native networking (eBPF policy
  enforcement), Workload Identity (no credential secrets in pods),
  private nodes + Cloud NAT, separate CPU and GPU node pools with
  autoscaling bounds mirroring the KEDA maxima.
- **Alternatives rejected:**
  - EKS/AKS — equivalent validity; GKE chosen for the L4 GPU price/
    performance point the capacity math uses.
  - Managed serverless (Cloud Run / Fargate) — GPU attachment and
    queue-driven scale-out are first-class needs here.
- **Consequences:**
  - (+) `terraform plan` review is the capacity-change gate — node maxima
    and cost ceilings change as reviewed diffs, never as console clicks.
  - (+) The cluster topology matches every claim made in the scaling and
    failure-mode documents.
  - (−) One cloud's shape in the reference deployment; the k8s manifests
    stay portable, and the Terraform is one honest concrete instance.
