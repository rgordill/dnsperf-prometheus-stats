"""Utility functions and constants for dnsperf parsing."""

from typing import Dict, List, Optional
from datetime import datetime
import sys

from .models import IntervalStats

# Regex pattern for extracting upper bound from bucket range (supports scientific notation)
BUCKET_UPPER_BOUND_PATTERN = r'([\d.eE+-]+)$'


def format_bound_for_label(value: float) -> str:
    """Format a float value as a string for Prometheus label.
    
    Always uses decimal notation (not scientific) to match dnsperf output format.
    Uses enough decimal places to preserve precision for very small values.
    """
    # Use decimal notation with sufficient precision (up to 9 decimal places)
    # Remove trailing zeros and decimal point if not needed
    return f"{value:.9f}".rstrip('0').rstrip('.')


def prepare_headers(remote_write_headers: Optional[List[str]]) -> Dict[str, str]:
    """Prepare headers dictionary from command-line arguments."""
    headers = {}
    if remote_write_headers:
        for header in remote_write_headers:
            if '=' in header:
                key, value = header.split('=', 1)
                headers[key] = value
    return headers


def build_interval_timestamps(intervals: List[IntervalStats]) -> Dict[int, datetime]:
    """Build a mapping from interval number to timestamp."""
    interval_timestamps = {}
    for interval in intervals:
        interval_timestamps[interval.interval_number] = interval.timestamp
    return interval_timestamps


def send_metrics_remote_write(remote_write_url: str, headers: Dict[str, str],
                               intervals: List[IntervalStats], instance_label: str, 
                               verbose: bool = False, dry_run: bool = False, debug_file: Optional[str] = None,
                               command_line_labels: Optional[Dict[str, str]] = None) -> None:
    """Send metrics via remote write endpoint.
    
    Args:
        remote_write_url: URL of the Prometheus remote write endpoint
        headers: HTTP headers to include in the request
        intervals: List of interval statistics to send
        instance_label: Value for the instance label added to all metrics
        verbose: Print verbose output for each metric
        dry_run: If True, process metrics but skip sending to endpoint
        debug_file: Optional path to save uncompressed payload data before compression
        command_line_labels: Optional dictionary of command line parameters to include in dnsperf_info metric
    """
    # Import here to avoid circular dependency
    from .remote_write import RemoteWriteClient
    
    if dry_run:
        print(f"\nDry-run mode: Processing metrics (not sending to {remote_write_url})...")
    else:
        print(f"\nSending metrics to {remote_write_url}...")
    
    client = RemoteWriteClient(remote_write_url, headers, instance_label, verbose)
    
    if client.send_metrics_from_intervals(intervals, dry_run=dry_run, debug_file=debug_file, command_line_labels=command_line_labels):
        if dry_run:
            print(f"Dry-run completed: Processed metrics for {len(intervals)} interval(s)")
        else:
            print(f"Successfully sent metrics for {len(intervals)} interval(s)")
    else:
        print("Failed to process/send metrics", file=sys.stderr)
        sys.exit(1)

