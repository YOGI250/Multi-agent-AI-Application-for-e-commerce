#!/bin/bash
# run-tests.sh — runs the correctness test suite with secrets from k8s

set -e

export DATABASE_URL=$(kubectl get secret ecommerce-secrets -n ecommerce -o jsonpath='{.data.database-url}' | base64 -d)
export GROQ_API_KEY=$(kubectl get secret ecommerce-secrets -n ecommerce -o jsonpath='{.data.groq-api-key}' | base64 -d)
export LANGFUSE_SECRET_KEY=$(kubectl get secret ecommerce-secrets -n ecommerce -o jsonpath='{.data.langfuse-secret-key}' | base64 -d)
export LANGFUSE_PUBLIC_KEY=$(kubectl get secret ecommerce-secrets -n ecommerce -o jsonpath='{.data.langfuse-public-key}' | base64 -d)
export LANGFUSE_HOST="https://cloud.langfuse.com"
export GOOGLE_CLIENT_ID=$(kubectl get secret ecommerce-secrets -n ecommerce -o jsonpath='{.data.google-client-id}' | base64 -d)
export GOOGLE_CLIENT_SECRET=$(kubectl get secret ecommerce-secrets -n ecommerce -o jsonpath='{.data.google-client-secret}' | base64 -d)
export GRAFANA_ADMIN_PASSWORD="admin"
export LANGFUSE_NEXTAUTH_SECRET="test"
export LANGFUSE_SALT="test"

source venv/bin/activate
python tests/test_system_correctness.py
