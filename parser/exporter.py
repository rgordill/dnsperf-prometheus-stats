"""Export dnsperf statistics as Prometheus metrics."""

import re
from typing import Dict, Optional

from prometheus_client import CollectorRegistry, Gauge

from .models import IntervalStats
from .utils import BUCKET_UPPER_BOUND_PATTERN, format_bound_for_label


class PrometheusMetricsExporter:
    """Export dnsperf statistics as Prometheus metrics."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Set up Prometheus metric definitions."""
        # Query metrics
        self.queries_sent = Gauge(
            'dnsperf_queries_sent_total',
            'Total number of DNS queries sent',
            [],
            registry=self.registry
        )
        self.queries_completed = Gauge(
            'dnsperf_queries_completed_total',
            'Total number of DNS queries completed',
            [],
            registry=self.registry
        )
        self.queries_lost = Gauge(
            'dnsperf_queries_lost_total',
            'Total number of DNS queries lost',
            [],
            registry=self.registry
        )
        
        # Response code metrics
        self.response_codes = Gauge(
            'dnsperf_response_codes_total',
            'Number of responses by code',
            ['code'],
            registry=self.registry
        )
        
        # Packet size metrics
        self.packet_size_request = Gauge(
            'dnsperf_packet_size_request_bytes',
            'Average request packet size in bytes',
            [],
            registry=self.registry
        )
        self.packet_size_response = Gauge(
            'dnsperf_packet_size_response_bytes',
            'Average response packet size in bytes',
            [],
            registry=self.registry
        )
        
        # Performance metrics
        self.run_time = Gauge(
            'dnsperf_run_time_seconds',
            'Run time for the interval',
            [],
            registry=self.registry
        )
        self.queries_per_second = Gauge(
            'dnsperf_queries_per_second',
            'Queries per second',
            [],
            registry=self.registry
        )
        
        # Latency metrics
        self.avg_latency = Gauge(
            'dnsperf_latency_seconds_avg',
            'Average latency in seconds',
            [],
            registry=self.registry
        )
        self.latency_stddev = Gauge(
            'dnsperf_latency_seconds_stddev',
            'Latency standard deviation in seconds',
            [],
            registry=self.registry
        )
        
        # Latency histogram buckets
        self.latency_bucket = Gauge(
            'dnsperf_latency_bucket',
            'Number of queries in latency bucket',
            ['le'],
            registry=self.registry
        )
        
        # Track which metrics belong to which interval for timestamp assignment
        self.metric_to_interval: Dict[str, int] = {}
    
    def export_interval(self, stats: IntervalStats):
        """Export metrics for a single interval."""
        # Query metrics
        self.queries_sent.set(stats.queries_sent)
        self.queries_completed.set(stats.queries_completed)
        self.queries_lost.set(stats.queries_lost)
        
        # Track metrics for timestamp assignment
        self.metric_to_interval['dnsperf_queries_sent_total{}'] = stats.interval_number
        self.metric_to_interval['dnsperf_queries_completed_total{}'] = stats.interval_number
        self.metric_to_interval['dnsperf_queries_lost_total{}'] = stats.interval_number
        
        # Response codes
        for code, count in stats.response_codes.items():
            self.response_codes.labels(code=code).set(count)
            self.metric_to_interval[f'dnsperf_response_codes_total{{code="{code}"}}'] = stats.interval_number
        
        # Packet sizes
        self.packet_size_request.set(stats.avg_packet_size_request)
        self.packet_size_response.set(stats.avg_packet_size_response)
        self.metric_to_interval['dnsperf_packet_size_request_bytes{}'] = stats.interval_number
        self.metric_to_interval['dnsperf_packet_size_response_bytes{}'] = stats.interval_number
        
        # Performance
        self.run_time.set(stats.run_time)
        self.queries_per_second.set(stats.queries_per_second)
        self.metric_to_interval['dnsperf_run_time_seconds{}'] = stats.interval_number
        self.metric_to_interval['dnsperf_queries_per_second{}'] = stats.interval_number
        
        # Latency
        self.avg_latency.set(stats.avg_latency)
        self.latency_stddev.set(stats.latency_stddev)
        self.metric_to_interval['dnsperf_latency_seconds_avg{}'] = stats.interval_number
        self.metric_to_interval['dnsperf_latency_seconds_stddev{}'] = stats.interval_number
        
        # Latency buckets (using upper bound as 'le' label)
        for bucket_range, count in stats.latency_buckets.items():
            # Extract upper bound from range (now "float-float" format)
            upper_str = bucket_range.split('-')[-1]
            try:
                upper_float = float(upper_str)
                # Format for label (preserves scientific notation for small values)
                upper_bound = format_bound_for_label(upper_float)
                self.latency_bucket.labels(le=upper_bound).set(count)
                self.metric_to_interval[f'dnsperf_latency_bucket{{le="{upper_bound}"}}'] = stats.interval_number
            except ValueError:
                # Fallback: try regex extraction if format is different
                upper_match = re.search(BUCKET_UPPER_BOUND_PATTERN, upper_str)
                if upper_match:
                    upper_bound = upper_match.group(1)
                    self.latency_bucket.labels(le=upper_bound).set(count)
                    self.metric_to_interval[f'dnsperf_latency_bucket{{le="{upper_bound}"}}'] = stats.interval_number

