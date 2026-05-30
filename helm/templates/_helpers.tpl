{{/*
Common labels applied to all resources.
*/}}
{{- define "ecommerce.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels used by Services and Deployments to match pods.
*/}}
{{- define "ecommerce.selectorLabels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Full image reference for the FastAPI app.
*/}}
{{- define "ecommerce.image" -}}
{{ .Values.image.repository }}:{{ .Values.image.tag }}
{{- end }}
