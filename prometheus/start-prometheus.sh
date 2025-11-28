#!/bin/bash
# Start Prometheus with remote write receiver enabled
# This allows Prometheus to receive metrics via the remote write protocol

CONFIG_FILE="${1:-prometheus.remote-write.yml}"
PROMETHEUS_VERSION="${PROMETHEUS_VERSION:-latest}"

echo "Starting Prometheus with remote write receiver enabled..."
echo "Config file: $CONFIG_FILE"
echo ""

# Check if running in Docker/Podman
if command -v podman &> /dev/null; then
    CONTAINER_CMD="podman"
elif command -v docker &> /dev/null; then
    CONTAINER_CMD="docker"
else
    echo "Error: Neither podman nor docker found. Please install one of them."
    exit 1
fi

# Get absolute path to config file
CONFIG_DIR="$(cd "$(dirname "$CONFIG_FILE")" && pwd)"
CONFIG_FILE_NAME="$(basename "$CONFIG_FILE")"
CONFIG_PATH="$CONFIG_DIR/$CONFIG_FILE_NAME"

echo "Using container command: $CONTAINER_CMD"
echo "Config path: $CONFIG_PATH"
echo ""

# Ensure config file is readable
chmod 644 "$CONFIG_PATH" 2>/dev/null || true

# Determine volume mount option based on container runtime and SELinux
VOLUME_OPTION=":ro"
if [ "$CONTAINER_CMD" = "podman" ]; then
    # Podman handles SELinux differently, try :Z first, fallback to :z or no suffix
    if [ -n "$(getenforce 2>/dev/null)" ] && [ "$(getenforce)" = "Enforcing" ]; then
        VOLUME_OPTION=":ro,Z"
    else
        VOLUME_OPTION=":ro"
    fi
elif [ "$CONTAINER_CMD" = "docker" ]; then
    # Docker with SELinux - use :Z
    if command -v getenforce &> /dev/null && [ "$(getenforce)" = "Enforcing" ]; then
        VOLUME_OPTION=":ro,Z"
    else
        VOLUME_OPTION=":ro"
    fi
fi

echo "Using volume option: $VOLUME_OPTION"
echo ""

# Stop and remove existing Prometheus container if it exists
CONTAINER_NAME="prometheus-remote-write"

# Check if container exists (running or stopped)
if $CONTAINER_CMD ps -a --format "{{.Names}}" 2>/dev/null | grep -q "^${CONTAINER_NAME}$"; then
    echo "Found existing Prometheus container '${CONTAINER_NAME}'"
    
    # Check if container is running and stop it
    if $CONTAINER_CMD ps --format "{{.Names}}" 2>/dev/null | grep -q "^${CONTAINER_NAME}$"; then
        echo "Stopping existing container..."
        $CONTAINER_CMD stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
        # Wait a moment for container to stop
        sleep 1
    fi
    
    # Remove the container (works for both stopped and running containers)
    echo "Removing existing container..."
    $CONTAINER_CMD rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    echo "Cleanup complete."
    echo ""
fi

# Start Prometheus container with remote write receiver enabled
# Note: Out-of-order sample ingestion is configured in prometheus.yml
# via storage.tsdb.out_of_order_time_window (requires Prometheus 2.39+)
$CONTAINER_CMD run -d \
    --name prometheus-remote-write \
    -p 9090:9090 \
    -v "$CONFIG_PATH:/etc/prometheus/prometheus.yml$VOLUME_OPTION" \
    quay.io/prometheus/prometheus:$PROMETHEUS_VERSION \
    --config.file=/etc/prometheus/prometheus.yml \
    --storage.tsdb.path=/prometheus \
    --web.console.libraries=/usr/share/prometheus/console_libraries \
    --web.console.templates=/usr/share/prometheus/consoles \
    --web.enable-lifecycle \
    --web.enable-remote-write-receiver

if [ $? -eq 0 ]; then
    echo ""
    echo "Prometheus started successfully!"
    echo "Remote write endpoint: http://localhost:9090/api/v1/write"
    echo "Web UI: http://localhost:9090"
    echo ""
    echo "To view logs: $CONTAINER_CMD logs -f prometheus-remote-write"
    echo "To stop: $CONTAINER_CMD stop prometheus-remote-write"
    echo "To remove: $CONTAINER_CMD rm prometheus-remote-write"
else
    echo "Failed to start Prometheus"
    exit 1
fi

