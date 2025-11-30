"""Data models for dnsperf statistics."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict


@dataclass
class IntervalStats:
    """Represents statistics for a single interval period."""
    interval_number: int
    timestamp: datetime  # Timestamp when this interval occurred
    queries_sent: int
    queries_completed: int
    queries_lost: int
    response_codes: Dict[str, int]
    avg_packet_size_request: int
    avg_packet_size_response: int
    run_time: float
    queries_per_second: float
    avg_latency: float
    latency_stddev: float
    latency_buckets: Dict[str, int]  # bucket_range -> count

