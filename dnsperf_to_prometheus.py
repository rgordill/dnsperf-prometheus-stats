#!/usr/bin/env python3
"""
Parse dnsperf statistics and convert them to Prometheus metrics,
then send them using Prometheus remote write.
"""

import re
import sys
import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram
from prometheus_client.exposition import generate_latest
import snappy
from google.protobuf.message import EncodeError

try:
    # Try to import prometheus remote write protobuf definitions
    # First try prometheus-remote-writer package (installed via requirements.txt)
    try:
        from prometheus_remote_writer.proto import remote_pb2 as prompb_pb2
    except ImportError:
        # Fallback: try alternative package name
        try:
            from prometheus_remote_write import prompb_pb2
        except ImportError:
            prompb_pb2 = None
except Exception as e:
    print(f"Warning: Could not import prometheus remote write protobuf: {e}", file=sys.stderr)
    prompb_pb2 = None


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


class DnsperfParser:
    """Parser for dnsperf statistics output."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.intervals: List[IntervalStats] = []
        self.start_time: Optional[datetime] = None
        
    def parse(self) -> List[IntervalStats]:
        """Parse the dnsperf stats file and return list of interval statistics."""
        with open(self.file_path, 'r') as f:
            content = f.read()
        
        # Parse the start timestamp
        start_time_match = re.search(r'\[Status\] Started at:\s+(.+)', content)
        if start_time_match:
            start_time_str = start_time_match.group(1).strip()
            # Parse format like "Thu Nov 27 15:58:14 2025"
            try:
                # Try common date formats
                for fmt in ['%a %b %d %H:%M:%S %Y', '%a %b %d %H:%M:%S %Y %Z']:
                    try:
                        self.start_time = datetime.strptime(start_time_str, fmt)
                        break
                    except ValueError:
                        continue
                if self.start_time is None:
                    print(f"Warning: Could not parse start time '{start_time_str}', using current time", file=sys.stderr)
                    self.start_time = datetime.now()
            except Exception as e:
                print(f"Warning: Error parsing start time: {e}, using current time", file=sys.stderr)
                self.start_time = datetime.now()
        else:
            print("Warning: Could not find start time, using current time", file=sys.stderr)
            self.start_time = datetime.now()
        
        # Find all interval statistics sections
        interval_pattern = r'Interval Statistics:\s*\n\s*\n(.*?)(?=\nInterval Statistics:|\n\[Status\]|\nStatistics:|\Z)'
        matches = re.finditer(interval_pattern, content, re.DOTALL)
        
        interval_num = 0
        previous_run_time = 0.0  # Track previous interval's run_time
        
        for match in matches:
            interval_num += 1
            interval_text = match.group(1)
            stats = self._parse_interval(interval_text, interval_num, previous_run_time)
            if stats:
                self.intervals.append(stats)
                # Update previous_run_time for next interval
                previous_run_time = stats.run_time
        
        return self.intervals
    
    def _parse_interval(self, text: str, interval_num: int, previous_run_time: float) -> Optional[IntervalStats]:
        """Parse a single interval statistics block.
        
        Args:
            text: The interval statistics text
            interval_num: The interval number
            previous_run_time: The run_time from the previous interval (0.0 for first interval)
        """
        try:
            # Extract queries sent
            queries_sent_match = re.search(r'Queries sent:\s+(\d+)', text)
            queries_sent = int(queries_sent_match.group(1)) if queries_sent_match else 0
            
            # Extract queries completed
            queries_completed_match = re.search(r'Queries completed:\s+(\d+)', text)
            queries_completed = int(queries_completed_match.group(1)) if queries_completed_match else 0
            
            # Extract queries lost
            queries_lost_match = re.search(r'Queries lost:\s+(\d+)', text)
            queries_lost = int(queries_lost_match.group(1)) if queries_lost_match else 0
            
            # Extract response codes
            response_codes = {}
            response_codes_match = re.search(r'Response codes:\s+(.+)', text)
            if response_codes_match:
                codes_text = response_codes_match.group(1)
                # Parse format like "NOERROR 101 (100.00%)"
                code_pattern = r'(\w+)\s+(\d+)'
                for code_match in re.finditer(code_pattern, codes_text):
                    code = code_match.group(1)
                    count = int(code_match.group(2))
                    response_codes[code] = count
            
            # Extract average packet size
            packet_size_match = re.search(r'Average packet size:\s+request\s+(\d+),\s+response\s+(\d+)', text)
            avg_packet_size_request = int(packet_size_match.group(1)) if packet_size_match else 0
            avg_packet_size_response = int(packet_size_match.group(2)) if packet_size_match else 0
            
            # Extract run time
            run_time_match = re.search(r'Run time \(s\):\s+([\d.]+)', text)
            run_time = float(run_time_match.group(1)) if run_time_match else 0.0
            
            # Extract queries per second
            qps_match = re.search(r'Queries per second:\s+([\d.]+)', text)
            queries_per_second = float(qps_match.group(1)) if qps_match else 0.0
            
            # Extract average latency
            avg_latency_match = re.search(r'Average Latency \(s\):\s+([\d.]+)', text)
            avg_latency = float(avg_latency_match.group(1)) if avg_latency_match else 0.0
            
            # Extract latency stddev
            latency_stddev_match = re.search(r'Latency StdDev \(s\):\s+([\d.]+)', text)
            latency_stddev = float(latency_stddev_match.group(1)) if latency_stddev_match else 0.0
            
            # Extract latency buckets
            latency_buckets = {}
            bucket_pattern = r'([\d.]+)\s+-\s+([\d.]+):\s+(\d+)'
            for bucket_match in re.finditer(bucket_pattern, text):
                lower = bucket_match.group(1)
                upper = bucket_match.group(2)
                count = int(bucket_match.group(3))
                bucket_range = f"{lower}-{upper}"
                latency_buckets[bucket_range] = count
            
            # Calculate timestamp: start_time + previous_run_time
            # The timestamp represents when this interval's statistics were collected,
            # which is at the end of the previous interval period
            if self.start_time is None:
                raise ValueError("start_time must be set before parsing intervals")
            timestamp = self.start_time + timedelta(seconds=previous_run_time)
            
            return IntervalStats(
                interval_number=interval_num,
                timestamp=timestamp,
                queries_sent=queries_sent,
                queries_completed=queries_completed,
                queries_lost=queries_lost,
                response_codes=response_codes,
                avg_packet_size_request=avg_packet_size_request,
                avg_packet_size_response=avg_packet_size_response,
                run_time=run_time,
                queries_per_second=queries_per_second,
                avg_latency=avg_latency,
                latency_stddev=latency_stddev,
                latency_buckets=latency_buckets
            )
        except Exception as e:
            print(f"Error parsing interval {interval_num}: {e}", file=sys.stderr)
            return None


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
            # Extract upper bound from range like "0.000144-0.000147"
            upper_match = re.search(r'([\d.]+)$', bucket_range.split('-')[-1])
            if upper_match:
                upper_bound = upper_match.group(1)
                self.latency_bucket.labels(le=upper_bound).set(count)
                self.metric_to_interval[f'dnsperf_latency_bucket{{le="{upper_bound}"}}'] = stats.interval_number


class RemoteWriteClient:
    """Client for sending Prometheus metrics via remote write."""
    
    def __init__(self, remote_write_url: str, headers: Optional[Dict[str, str]] = None, interval_timestamps: Optional[Dict[int, datetime]] = None, instance_label: str = 'dnsperf'):
        self.remote_write_url = remote_write_url
        self.headers = headers or {}
        self.headers.setdefault('Content-Type', 'application/x-protobuf')
        self.headers.setdefault('Content-Encoding', 'snappy')
        self.interval_timestamps = interval_timestamps or {}  # interval_number -> timestamp
        self.instance_label = instance_label  # Value for the instance label
    
    def send_metrics_from_intervals(self, intervals: List[IntervalStats]) -> bool:
        """Send metrics directly from interval statistics to remote write endpoint."""
        try:
            # Convert intervals directly to Prometheus remote write format
            write_request = self._convert_intervals_to_remote_write(intervals)
            
            if write_request is None:
                print("Error: Could not convert intervals to remote write format", file=sys.stderr)
                return False
            
            # Diagnostic information
            num_timeseries = len(write_request.timeseries)
            total_samples = sum(len(ts.samples) for ts in write_request.timeseries)
            print(f"Prepared {num_timeseries} time series with {total_samples} total samples", file=sys.stderr)
            
            # Get timestamp range
            all_timestamps = []
            for ts in write_request.timeseries:
                for sample in ts.samples:
                    all_timestamps.append(sample.timestamp)
            if all_timestamps:
                min_ts = min(all_timestamps) / 1000.0
                max_ts = max(all_timestamps) / 1000.0
                print(f"Timestamp range: {datetime.fromtimestamp(min_ts)} to {datetime.fromtimestamp(max_ts)}", file=sys.stderr)
            
            # Serialize and compress
            data = write_request.SerializeToString()
            compressed_data = snappy.compress(data)
            print(f"Sending {len(compressed_data)} bytes (uncompressed: {len(data)} bytes)", file=sys.stderr)
            
            # Send to remote write endpoint
            response = requests.post(
                self.remote_write_url,
                data=compressed_data,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200 or response.status_code == 204:
                print(f"Successfully sent metrics (status {response.status_code})", file=sys.stderr)
                return True
            else:
                print(f"Error sending metrics: {response.status_code} - {response.text}", file=sys.stderr)
                print(f"Response headers: {dict(response.headers)}", file=sys.stderr)
                return False
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error: Could not connect to {self.remote_write_url}", file=sys.stderr)
            print(f"  Make sure Prometheus is running and the remote write receiver is enabled", file=sys.stderr)
            print(f"  Start Prometheus with: --web.enable-remote-write-receiver", file=sys.stderr)
            return False
        except Exception as e:
            print(f"Error in remote write: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return False
    
    def send_metrics(self, registry: CollectorRegistry, metric_to_interval: Dict[str, int]) -> bool:
        """Send metrics from registry to remote write endpoint."""
        try:
            # Convert metrics to Prometheus remote write format
            write_request = self._convert_to_remote_write(registry, self.interval_timestamps, metric_to_interval)
            
            if write_request is None:
                print("Error: Could not convert metrics to remote write format", file=sys.stderr)
                return False
            
            # Serialize and compress
            data = write_request.SerializeToString()
            compressed_data = snappy.compress(data)
            
            # Send to remote write endpoint
            response = requests.post(
                self.remote_write_url,
                data=compressed_data,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200 or response.status_code == 204:
                return True
            else:
                print(f"Error sending metrics: {response.status_code} - {response.text}", file=sys.stderr)
                return False
        except Exception as e:
            print(f"Error in remote write: {e}", file=sys.stderr)
            # Fallback: try text format if remote write fails
            print("Attempting fallback to text format...", file=sys.stderr)
            try:
                metrics_text = generate_latest(registry)
                response = requests.post(
                    self.remote_write_url,
                    data=metrics_text,
                    headers={'Content-Type': 'text/plain'},
                    timeout=30
                )
                if response.status_code == 200 or response.status_code == 204:
                    print("Successfully sent metrics using text format fallback", file=sys.stderr)
                    return True
            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}", file=sys.stderr)
            return False
    
    def _convert_to_remote_write(self, registry: CollectorRegistry, interval_timestamps: Dict[int, datetime], metric_to_interval: Dict[str, int]):
        """Convert Prometheus registry to remote write format."""
        if prompb_pb2 is None:
            # Try to create protobuf structure manually or return None
            return None
        
        write_request = prompb_pb2.WriteRequest()
        
        # Get all metrics from registry
        metrics_text = generate_latest(registry).decode('utf-8')
        
        # Parse text format and convert to TimeSeries
        for line in metrics_text.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse metric line
            # Format: metric_name{labels} value [timestamp]
            parts = line.split()
            if len(parts) < 2:
                continue
            
            metric_part = parts[0]
            try:
                value = float(parts[1])
            except ValueError:
                continue
            
            # Parse metric name and labels
            if '{' in metric_part:
                name = metric_part.split('{')[0]
                labels_text = metric_part.split('{')[1].rstrip('}')
                labels = {}
                if labels_text:
                    # Parse labels like: label1="value1",label2="value2"
                    for label_pair in labels_text.split(','):
                        label_pair = label_pair.strip()
                        if '=' in label_pair:
                            key, val = label_pair.split('=', 1)
                            # Remove quotes if present
                            val = val.strip('"').strip("'")
                            # Skip interval label - we don't want it in the output
                            if key != 'interval':
                                labels[key] = val
            else:
                name = metric_part
                labels = {}
            
            # Build metric key for lookup (format: name{label1="value1",label2="value2"})
            label_parts = [f'{k}="{v}"' for k, v in sorted(labels.items())]
            metric_key = f'{name}{{{",".join(label_parts)}}}'
            
            # Determine timestamp: use interval timestamp from mapping
            timestamp_ms = int(datetime.now().timestamp() * 1000)  # default to now
            if metric_key in metric_to_interval:
                interval_num = metric_to_interval[metric_key]
                if interval_num in interval_timestamps:
                    timestamp_ms = int(interval_timestamps[interval_num].timestamp() * 1000)
            
            # Create TimeSeries for this metric
            time_series = write_request.timeseries.add()
            
            # Add __name__ label
            label = time_series.labels.add()
            label.name = '__name__'
            label.value = name
            
            # Add other labels
            for key, val in labels.items():
                label = time_series.labels.add()
                label.name = key
                label.value = val
            
            # Add sample
            sample = time_series.samples.add()
            sample.value = value
            sample.timestamp = timestamp_ms
        
        return write_request
    
    def _convert_intervals_to_remote_write(self, intervals: List[IntervalStats]):
        """Convert interval statistics directly to remote write format."""
        if prompb_pb2 is None:
            return None
        
        write_request = prompb_pb2.WriteRequest()
        
        # Group samples by their unique label combination (metric name + labels)
        # This is required by Prometheus remote write format
        time_series_map: Dict[tuple, Any] = {}  # (metric_name, sorted_labels_tuple) -> TimeSeries
        
        # Track cumulative totals for _total metrics (counters)
        cumulative_queries_sent = 0
        cumulative_queries_completed = 0
        cumulative_queries_lost = 0
        cumulative_response_codes: Dict[str, int] = {}  # code -> cumulative count
        cumulative_latency_sum = 0.0  # Cumulative sum of all latency values
        cumulative_latency_count = 0  # Cumulative count of latency observations (for histogram)
        cumulative_bucket_counts: Dict[str, int] = {}  # bucket_upper_bound -> cumulative count
        
        # First pass: collect all unique bucket upper bounds across all intervals
        all_bucket_bounds = set()
        for stats in intervals:
            for bucket_range, count in stats.latency_buckets.items():
                upper_match = re.search(r'([\d.]+)$', bucket_range.split('-')[-1])
                if upper_match:
                    upper_bound = upper_match.group(1)
                    all_bucket_bounds.add(upper_bound)
        
        # Sort bucket bounds numerically
        sorted_bucket_bounds = sorted([float(bound) for bound in all_bucket_bounds])
        sorted_bucket_bounds_str = [str(bound) for bound in sorted_bucket_bounds]
        
        for stats in intervals:
            timestamp_ms = int(stats.timestamp.timestamp() * 1000)
            
            # Accumulate _total metrics (counters)
            cumulative_queries_sent += stats.queries_sent
            cumulative_queries_completed += stats.queries_completed
            cumulative_queries_lost += stats.queries_lost
            
            # Accumulate response codes
            for code, count in stats.response_codes.items():
                if code not in cumulative_response_codes:
                    cumulative_response_codes[code] = 0
                cumulative_response_codes[code] += count
            
            # Query metrics (send cumulative totals)
            self._add_sample_to_map(time_series_map, 'dnsperf_queries_sent_total', {}, cumulative_queries_sent, timestamp_ms)
            self._add_sample_to_map(time_series_map, 'dnsperf_queries_completed_total', {}, cumulative_queries_completed, timestamp_ms)
            self._add_sample_to_map(time_series_map, 'dnsperf_queries_lost_total', {}, cumulative_queries_lost, timestamp_ms)
            
            # Response codes (send cumulative totals)
            for code, cumulative_count in cumulative_response_codes.items():
                self._add_sample_to_map(time_series_map, 'dnsperf_response_codes_total', {'code': code}, cumulative_count, timestamp_ms)
            
            # Packet sizes
            self._add_sample_to_map(time_series_map, 'dnsperf_packet_size_request_bytes', {}, stats.avg_packet_size_request, timestamp_ms)
            self._add_sample_to_map(time_series_map, 'dnsperf_packet_size_response_bytes', {}, stats.avg_packet_size_response, timestamp_ms)
            
            # Performance
            self._add_sample_to_map(time_series_map, 'dnsperf_run_time_seconds', {}, stats.run_time, timestamp_ms)
            self._add_sample_to_map(time_series_map, 'dnsperf_queries_per_second', {}, stats.queries_per_second, timestamp_ms)
            
            # Latency
            self._add_sample_to_map(time_series_map, 'dnsperf_latency_seconds_avg', {}, stats.avg_latency, timestamp_ms)
            self._add_sample_to_map(time_series_map, 'dnsperf_latency_seconds_stddev', {}, stats.latency_stddev, timestamp_ms)
            
            # Calculate latency sum from buckets (use midpoint of each bucket range)
            # Also count total observations for histogram _count metric
            interval_latency_sum = 0.0
            interval_latency_count = 0
            for bucket_range, count in stats.latency_buckets.items():
                interval_latency_count += count  # Count observations in this interval
                # Parse bucket range (format: "lower-upper")
                bucket_parts = bucket_range.split('-')
                if len(bucket_parts) == 2:
                    try:
                        lower = float(bucket_parts[0])
                        upper = float(bucket_parts[1])
                        # Use midpoint of bucket range multiplied by count
                        midpoint = (lower + upper) / 2.0
                        interval_latency_sum += midpoint * count
                    except ValueError:
                        # Skip invalid bucket ranges
                        continue
            
            # Accumulate latency sum and count
            cumulative_latency_sum += interval_latency_sum
            cumulative_latency_count += interval_latency_count
            
            # Add latency sum metric (cumulative counter)
            self._add_sample_to_map(time_series_map, 'dnsperf_latency_seconds_sum', {}, cumulative_latency_sum, timestamp_ms)
            
            # Add latency count metric (cumulative counter) - required for histogram
            self._add_sample_to_map(time_series_map, 'dnsperf_latency_seconds_count', {}, cumulative_latency_count, timestamp_ms)
            
            # Latency buckets - make them cumulative (histogram buckets must be cumulative)
            # dnsperf buckets are per-bucket counts (not cumulative), we need to make them cumulative
            # Step 1: Collect per-bucket counts for this interval
            interval_per_bucket_counts: Dict[str, int] = {}  # upper_bound -> per-bucket count
            for bucket_range, count in stats.latency_buckets.items():
                upper_match = re.search(r'([\d.]+)$', bucket_range.split('-')[-1])
                if upper_match:
                    upper_bound = upper_match.group(1)
                    interval_per_bucket_counts[upper_bound] = count
            
            # Step 2: Convert per-bucket counts to cumulative counts within this interval
            cumulative_interval_counts: Dict[str, int] = {}
            running_cumulative = 0
            for bound_float in sorted_bucket_bounds:
                bound_str = str(bound_float)
                # Add per-bucket count from this interval
                running_cumulative += interval_per_bucket_counts.get(bound_str, 0)
                cumulative_interval_counts[bound_str] = running_cumulative
            
            # Step 3: Accumulate cumulative counts across intervals
            for upper_bound in sorted_bucket_bounds_str:
                interval_cumulative = cumulative_interval_counts.get(upper_bound, 0)
                if upper_bound not in cumulative_bucket_counts:
                    cumulative_bucket_counts[upper_bound] = 0
                # Add this interval's cumulative value to the total
                cumulative_bucket_counts[upper_bound] += interval_cumulative
            
            # Step 4: Emit all buckets with cumulative counts
            for upper_bound in sorted_bucket_bounds_str:
                cumulative_count = cumulative_bucket_counts.get(upper_bound, 0)
                self._add_sample_to_map(time_series_map, 'dnsperf_latency_bucket', {'le': upper_bound}, cumulative_count, timestamp_ms)
            
            # Step 5: Add +Inf bucket (required for histogram_quantile) - equals the total count
            self._add_sample_to_map(time_series_map, 'dnsperf_latency_bucket', {'le': '+Inf'}, cumulative_latency_count, timestamp_ms)
        
        # Add all grouped TimeSeries to the write request
        # Sort samples by timestamp within each TimeSeries to ensure chronological order
        for time_series in time_series_map.values():
            # Sort samples by timestamp (Prometheus requires strictly monotonically increasing timestamps)
            samples_list = [(s.timestamp, s.value) for s in time_series.samples]
            samples_list.sort(key=lambda x: x[0])  # Sort by timestamp
            
            # Remove duplicate timestamps (keep the last value for each timestamp)
            # Prometheus requires strictly increasing timestamps (no duplicates)
            deduplicated_samples = []
            last_timestamp = None
            for timestamp, value in samples_list:
                if timestamp == last_timestamp:
                    # Update the last sample's value (keep most recent)
                    deduplicated_samples[-1] = (timestamp, value)
                elif timestamp > last_timestamp if last_timestamp is not None else True:
                    deduplicated_samples.append((timestamp, value))
                    last_timestamp = timestamp
                else:
                    # This shouldn't happen after sorting, but handle it just in case
                    print(f"Warning: Found out-of-order timestamp {timestamp} after {last_timestamp}", file=sys.stderr)
                    # Skip this sample to maintain strict ordering
                    continue
            
            # Validate: ensure strictly increasing timestamps
            if len(deduplicated_samples) > 1:
                timestamps = [ts for ts, _ in deduplicated_samples]
                for i in range(1, len(timestamps)):
                    if timestamps[i] <= timestamps[i-1]:
                        metric_name = None
                        for label in time_series.labels:
                            if label.name == '__name__':
                                metric_name = label.value
                                break
                        print(f"Error: TimeSeries {metric_name} has non-strictly-increasing timestamps at index {i}: {timestamps[i-1:i+1]}", file=sys.stderr)
                        # Fix by removing the problematic sample
                        deduplicated_samples.pop(i)
                        break
            
            # Clear existing samples and add sorted, deduplicated ones
            time_series.ClearField('samples')
            for timestamp, value in deduplicated_samples:
                sample = time_series.samples.add()
                sample.timestamp = timestamp
                sample.value = value
            
            # Only add TimeSeries that have at least one sample
            if len(time_series.samples) > 0:
                # Copy the TimeSeries to the write request
                new_ts = write_request.timeseries.add()
                new_ts.CopyFrom(time_series)
        
        return write_request
    
    def _add_sample_to_map(self, time_series_map: Dict[tuple, Any], metric_name: str, labels: Dict[str, str], value: float, timestamp_ms: int):
        """Add a sample to the time series map, grouping by metric name and labels."""
        # Add instance label to all metrics
        labels_with_instance = labels.copy()
        labels_with_instance['instance'] = self.instance_label
        
        # Create a unique key from metric name and sorted labels
        sorted_labels = tuple(sorted(labels_with_instance.items()))
        key = (metric_name, sorted_labels)
        
        # Get or create the TimeSeries for this key
        if key not in time_series_map:
            # Import types_pb2 for TimeSeries
            from prometheus_remote_writer.proto import types_pb2
            time_series = types_pb2.TimeSeries()
            
            # Add __name__ label
            label = time_series.labels.add()
            label.name = '__name__'
            label.value = metric_name
            
            # Add other labels (including instance label)
            for key_name, val in labels_with_instance.items():
                label = time_series.labels.add()
                label.name = key_name
                label.value = str(val)
            
            time_series_map[key] = time_series
        
        # Add sample to the existing TimeSeries
        sample = time_series_map[key].samples.add()
        sample.value = value
        sample.timestamp = timestamp_ms
    
    def _add_sample(self, write_request, metric_name: str, labels: Dict[str, str], value: float, timestamp_ms: int):
        """Add a sample to the write request (deprecated - use _add_sample_to_map instead)."""
        time_series = write_request.timeseries.add()
        
        # Add __name__ label
        label = time_series.labels.add()
        label.name = '__name__'
        label.value = metric_name
        
        # Add other labels
        for key, val in labels.items():
            label = time_series.labels.add()
            label.name = key
            label.value = str(val)
        
        # Add sample
        sample = time_series.samples.add()
        sample.value = value
        sample.timestamp = timestamp_ms


def main():
    parser = argparse.ArgumentParser(
        description='Parse dnsperf statistics and send to Prometheus via remote write'
    )
    parser.add_argument(
        'input_file',
        help='Path to dnsperf statistics file'
    )
    parser.add_argument(
        '--remote-write-url',
        required=True,
        help='Prometheus remote write endpoint URL'
    )
    parser.add_argument(
        '--remote-write-header',
        action='append',
        help='Additional header for remote write (format: Key=Value)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Parse and display metrics without sending'
    )
    parser.add_argument(
        '--instance-label',
        default='dnsperf',
        help='Value for the instance label added to all metrics (default: dnsperf)'
    )
    
    args = parser.parse_args()
    
    # Parse dnsperf stats
    print(f"Parsing dnsperf statistics from {args.input_file}...")
    dnsperf_parser = DnsperfParser(args.input_file)
    intervals = dnsperf_parser.parse()
    
    if not intervals:
        print("No interval statistics found in file.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(intervals)} interval(s)")
    
    # Build interval to timestamp mapping
    interval_timestamps = {}
    for interval in intervals:
        interval_timestamps[interval.interval_number] = interval.timestamp
    
    if args.dry_run:
        # For dry-run, create a registry to display metrics
        print("Creating Prometheus metrics...")
        registry = CollectorRegistry()
        exporter = PrometheusMetricsExporter(registry)
        
        for interval in intervals:
            exporter.export_interval(interval)
            print(f"  Exported metrics for interval {interval.interval_number} (timestamp: {interval.timestamp})")
        
        print("\nMetrics (text format - note: only last interval's values shown due to label removal):")
        print(generate_latest(registry).decode('utf-8'))
        print("\nInterval timestamps:")
        for interval_num, ts in sorted(interval_timestamps.items()):
            print(f"  Interval {interval_num}: {ts} ({int(ts.timestamp() * 1000)} ms)")
    else:
        # Prepare remote write headers
        headers = {}
        if args.remote_write_header:
            for header in args.remote_write_header:
                if '=' in header:
                    key, value = header.split('=', 1)
                    headers[key] = value
        
        # Send metrics via remote write
        print(f"\nSending metrics to {args.remote_write_url}...")
        client = RemoteWriteClient(args.remote_write_url, headers, interval_timestamps, args.instance_label)
        
        # Create TimeSeries directly from intervals (not using registry to avoid overwrites)
        if client.send_metrics_from_intervals(intervals):
            print(f"Successfully sent metrics for {len(intervals)} interval(s)")
        else:
            print("Failed to send metrics", file=sys.stderr)
            sys.exit(1)


if __name__ == '__main__':
    main()

