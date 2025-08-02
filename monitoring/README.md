# Monitoring & Observability for liquid-audio-nets

This directory contains monitoring and observability configurations for the liquid-audio-nets project.

## 🔍 Overview

The monitoring stack provides comprehensive observability for:
- **Performance Metrics**: Inference latency, throughput, power consumption
- **System Health**: CPU, memory, temperature monitoring
- **Model Quality**: Accuracy tracking and degradation detection
- **Hardware Monitoring**: Embedded device health and connectivity
- **Training Metrics**: Loss, accuracy, convergence monitoring

## 📊 Components

### Prometheus
- **Config**: `prometheus.yml`
- **Alerts**: `rules/alerts.yml`
- **Port**: 9090
- **Purpose**: Metrics collection and alerting

### Grafana
- **Config**: `grafana/provisioning/`
- **Dashboards**: `grafana/dashboards/`
- **Port**: 3000
- **Credentials**: admin/admin (change in production)

### Alertmanager
- **Config**: `alertmanager.yml`
- **Port**: 9093
- **Purpose**: Alert routing and notification

### OpenTelemetry Collector
- **Config**: `otel/collector.yml`
- **Ports**: 4317 (gRPC), 4318 (HTTP)
- **Purpose**: Unified telemetry collection

## 🚀 Quick Start

### Start Full Monitoring Stack
```bash
cd monitoring
docker-compose -f docker-compose.monitoring.yml up -d
```

### Access Dashboards
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Alertmanager**: http://localhost:9093

### View Metrics
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Query metrics
curl 'http://localhost:9090/api/v1/query?query=inference_latency_ms'
```

## 🎯 Key Metrics

### Performance Metrics
- `inference_latency_ms`: Model inference time
- `audio_samples_processed_per_second`: Throughput
- `power_consumption_mw`: Power usage
- `memory_usage_mb`: Memory consumption

### Quality Metrics
- `model_accuracy`: Current model accuracy
- `validation_accuracy`: Validation performance
- `training_loss`: Training loss value

### Hardware Metrics
- `device_temperature_celsius`: Device temperature
- `cpu_usage_percent`: CPU utilization
- `battery_level_percent`: Battery status

## ⚠️ Alerts

### Critical Alerts
- **DeviceOffline**: Embedded device disconnected
- **LowThroughput**: Below real-time processing
- **LowModelAccuracy**: Accuracy below 90%

### Warning Alerts
- **HighInferenceLatency**: Latency >25ms
- **HighPowerConsumption**: Power >3mW
- **HighMemoryUsage**: Memory >10MB

## 🔧 Configuration

### Custom Metrics
Add application metrics to your code:

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
inference_counter = Counter('inferences_total', 'Total inferences')
latency_histogram = Histogram('inference_latency_seconds', 'Inference latency')
power_gauge = Gauge('power_consumption_watts', 'Power consumption')

# Use in code
inference_counter.inc()
with latency_histogram.time():
    result = model.predict(input_data)
power_gauge.set(measure_power())
```

### Custom Alerts
Add to `rules/alerts.yml`:

```yaml
- alert: CustomAlert
  expr: your_metric > threshold
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Custom alert triggered"
    description: "Your metric is {{ $value }}"
```

### Environment Variables
Configure via environment:

```bash
export PROMETHEUS_URL=http://localhost:9090
export GRAFANA_URL=http://localhost:3000
export ALERT_WEBHOOK_URL=http://your-webhook.com
```

## 📈 Dashboards

### Available Dashboards
1. **System Overview**: Overall system health
2. **Performance**: Latency, throughput, power
3. **Model Quality**: Accuracy and training metrics
4. **Hardware**: Embedded device monitoring

### Creating Custom Dashboards
1. Access Grafana at http://localhost:3000
2. Create new dashboard
3. Add panels with PromQL queries
4. Save dashboard JSON to `grafana/dashboards/`

## 🔐 Security

### Production Considerations
- Change default Grafana password
- Enable authentication for Prometheus
- Use TLS for all connections
- Restrict network access
- Configure proper RBAC

### Secrets Management
```bash
# Use environment variables for sensitive data
export SLACK_WEBHOOK_URL="your-slack-webhook"
export SMTP_PASSWORD="your-smtp-password"
```

## 🧪 Testing

### Health Checks
```bash
# Check all services
./scripts/check_monitoring_health.sh

# Test alerting
./scripts/test_alerts.sh
```

### Metric Validation
```bash
# Validate metrics endpoint
curl http://localhost:8080/metrics

# Check metric format
promtool query instant 'up'
```

## 📚 Integration

### With CI/CD
Add to your pipeline:

```yaml
- name: Start monitoring
  run: docker-compose -f monitoring/docker-compose.monitoring.yml up -d

- name: Run tests with metrics
  run: PROMETHEUS_URL=http://localhost:9090 pytest tests/

- name: Export test metrics
  run: ./scripts/export_test_metrics.sh
```

### With Applications
```python
# Initialize OpenTelemetry
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317")
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)
```

## 🔍 Troubleshooting

### Common Issues

**Prometheus not scraping metrics**
- Check target endpoints are accessible
- Verify metrics format with `/metrics` endpoint
- Check firewall/network connectivity

**Grafana not showing data**
- Verify Prometheus datasource configuration
- Check query syntax in panels
- Ensure time ranges are correct

**Alerts not firing**
- Check alert rule syntax
- Verify metric data availability
- Test alertmanager configuration

### Debugging Commands
```bash
# Check Prometheus config
promtool check config prometheus.yml

# Validate alert rules
promtool check rules rules/alerts.yml

# Test alertmanager config
amtool config show --config.file=alertmanager.yml
```

## 📞 Support

For monitoring-related issues:
1. Check the troubleshooting section above
2. Review logs: `docker-compose -f docker-compose.monitoring.yml logs`
3. Open an issue with the `monitoring` label
4. Contact the DevOps team

## 🔗 Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Alertmanager Guide](https://prometheus.io/docs/alerting/latest/alertmanager/)