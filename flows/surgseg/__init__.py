"""Shared platform code used by the Metaflow flows.

It lives inside flows/ on purpose: Metaflow packages the flow's folder and
ships it to every Kubernetes pod, so code here is available in the cluster.
"""