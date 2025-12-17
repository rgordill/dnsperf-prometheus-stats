#!/bin/bash
# Start Prometheus with remote write receiver enabled
# This allows Prometheus to receive metrics via the remote write protocol

set -e

# Default values
CONFIG_FILE="prometheus.remote-write.yml"
DATA_DIR=""
PROMETHEUS_VERSION="${PROMETHEUS_VERSION:-latest}"

# Usage function
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Start Prometheus with remote write receiver enabled.

Options:
    --config=FILE       Path to Prometheus config file (default: prometheus.remote-write.yml)
    --data-dir=DIR      Path to persistent data directory (optional)
                        If not specified, data is stored inside the container
    --version=VERSION   Prometheus version to use (default: latest)
    --help              Show this help message and exit

Environment Variables:
    PROMETHEUS_VERSION  Prometheus container image version (default: latest)

Examples:
    $(basename "$0")
    $(basename "$0") --config=/path/to/prometheus.yml
    $(basename "$0") --config=myconfig.yml --data-dir=/var/lib/prometheus
    $(basename "$0") --data-dir=./prometheus-data --version=v2.47.0

EOF
    exit "${1:-0}"
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config=*)
            CONFIG_FILE="${1#*=}"
            shift
            ;;
        --data-dir=*)
            DATA_DIR="${1#*=}"
            shift
            ;;
        --version=*)
            PROMETHEUS_VERSION="${1#*=}"
            shift
            ;;
        --help|-h)
            usage 0
            ;;
        *)
            echo "Error: Unknown option: $1" >&2
            echo "" >&2
            usage 1
            ;;
    esac
done

echo "Starting Prometheus with remote write receiver enabled..."
echo "Config file: $CONFIG_FILE"
if [ -n "$DATA_DIR" ]; then
    echo "Data directory: $DATA_DIR"
fi
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

# Build volume arguments
VOLUME_ARGS=(-v "$CONFIG_PATH:/etc/prometheus/prometheus.yml$VOLUME_OPTION")

# Add data directory volume if specified
if [ -n "$DATA_DIR" ]; then
    # Get absolute path to data directory
    if [[ "$DATA_DIR" != /* ]]; then
        DATA_DIR="$(pwd)/$DATA_DIR"
    fi
    
    # Create data directory if it doesn't exist
    if [ ! -d "$DATA_DIR" ]; then
        echo "Creating data directory: $DATA_DIR"
        mkdir -p "$DATA_DIR"
    fi
    
    # Determine data volume option (needs write access)
    # Podman: use :U to auto-chown to match container user (nobody/65534)
    # Also use :Z for SELinux if enforcing
    DATA_VOLUME_OPTION=":rw"
    if [ "$CONTAINER_CMD" = "podman" ]; then
        if [ -n "$(getenforce 2>/dev/null)" ] && [ "$(getenforce)" = "Enforcing" ]; then
            DATA_VOLUME_OPTION=":rw,Z,U"
        else
            DATA_VOLUME_OPTION=":rw,U"
        fi
    elif [ "$CONTAINER_CMD" = "docker" ]; then
        # Docker: need to manually set ownership to 65534:65534 (nobody)
        echo "Setting ownership of data directory for Docker..."
        chown -R 65534:65534 "$DATA_DIR" 2>/dev/null || sudo chown -R 65534:65534 "$DATA_DIR"
        if command -v getenforce &> /dev/null && [ "$(getenforce)" = "Enforcing" ]; then
            DATA_VOLUME_OPTION=":rw,Z"
        fi
    fi
    
    echo "Using data directory: $DATA_DIR"
    echo "Data volume option: $DATA_VOLUME_OPTION"
    echo ""
    
    VOLUME_ARGS+=(-v "$DATA_DIR:/prometheus$DATA_VOLUME_OPTION")
fi

$CONTAINER_CMD run -d \
    --name prometheus-remote-write \
    -p 9090:9090 \
    "${VOLUME_ARGS[@]}" \
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

