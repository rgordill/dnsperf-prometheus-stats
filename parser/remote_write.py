"""Client for sending Prometheus metrics via remote write."""

import re
import sys
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import requests
import snappy
from google.protobuf.json_format import MessageToJson

from prometheus_remote_writer.proto import remote_pb2 as prompb_pb2
from prometheus_remote_writer.proto import types_pb2

from .models import IntervalStats
from .utils import BUCKET_UPPER_BOUND_PATTERN, format_bound_for_label


class RemoteWriteClient:
    """Client for sending Prometheus metrics via remote write."""
    
    def __init__(self, remote_write_url: str, headers: Optional[Dict[str, str]] = None, instance_label: str = 'dnsperf', verbose: bool = False):
        self.remote_write_url = remote_write_url
        self.headers = headers or {}
        self.headers.setdefault('Content-Type', 'application/x-protobuf')
        self.headers.setdefault('Content-Encoding', 'snappy')
        self.instance_label = instance_label  # Value for the instance label
        self.verbose = verbose
    
    def send_metrics_from_intervals(self, intervals: List[IntervalStats], dry_run: bool = False, debug_file: Optional[str] = None, command_line_labels: Optional[Dict[str, str]] = None) -> bool:
        """Send metrics directly from interval statistics to remote write endpoint.
        
        Args:
            intervals: List of interval statistics to send
            dry_run: If True, process metrics but skip sending to endpoint
            debug_file: Optional path to save uncompressed payload data before compression
            command_line_labels: Optional dictionary of command line parameters to include in dnsperf_info metric
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Log interval information before conversion
            if intervals:
                first_interval = intervals[0]
                last_interval = intervals[-1]
                print(f"Processing {len(intervals)} intervals", file=sys.stderr)
                print(f"  First interval: timestamp={first_interval.timestamp}, run_time={first_interval.run_time:.6f}s", file=sys.stderr)
                print(f"  Last interval: timestamp={last_interval.timestamp}, run_time={last_interval.run_time:.6f}s", file=sys.stderr)
                
                # Calculate expected duration (time span should be last_run_time - first_run_time)
                expected_duration = last_interval.run_time - first_interval.run_time
                time_span = (last_interval.timestamp - first_interval.timestamp).total_seconds()
                print(f"  Expected time span: {expected_duration:.6f}s (last_run_time - first_run_time), Actual time span: {time_span:.6f}s", file=sys.stderr)
                
                # Warn if time span is significantly different from expected duration
                if abs(time_span - expected_duration) > 1.0:
                    print(f"  WARNING: Time span ({time_span:.6f}s) differs significantly from expected ({expected_duration:.6f}s)", file=sys.stderr)
            
            # Convert intervals directly to Prometheus remote write format
            write_request = self._convert_intervals_to_remote_write(intervals, command_line_labels=command_line_labels)
            
            if write_request is None:
                print("Error: Could not convert intervals to remote write format", file=sys.stderr)
                return False
            
            # Diagnostic information
            num_timeseries = len(write_request.timeseries)
            total_samples = sum(len(ts.samples) for ts in write_request.timeseries)
            print(f"Prepared {num_timeseries} time series with {total_samples} total samples", file=sys.stderr)
            
            # Get timestamp range and validate
            all_timestamps = []
            for ts in write_request.timeseries:
                for sample in ts.samples:
                    all_timestamps.append(sample.timestamp)
            
            if all_timestamps:
                min_ts = min(all_timestamps) / 1000.0
                max_ts = max(all_timestamps) / 1000.0
                min_dt = datetime.fromtimestamp(min_ts)
                max_dt = datetime.fromtimestamp(max_ts)
                print(f"Timestamp range: {min_dt} (Unix: {int(min_ts)}) to {max_dt} (Unix: {int(max_ts)})", file=sys.stderr)
                
                # Validate timestamps against intervals
                if intervals:
                    first_ts = int(intervals[0].timestamp.timestamp())
                    last_ts = int(intervals[-1].timestamp.timestamp())
                    expected_min = first_ts
                    expected_max = last_ts
                    
                    # Check for timestamps outside expected range
                    outliers = []
                    for ts_ms in all_timestamps:
                        ts_s = int(ts_ms / 1000)
                        if ts_s < expected_min - 10 or ts_s > expected_max + 10:  # 10 second tolerance
                            if ts_s not in outliers:
                                outliers.append(ts_s)
                    
                    if outliers:
                        print(f"  WARNING: Found {len(outliers)} outlier timestamp(s) outside expected range:", file=sys.stderr)
                        print(f"    Expected range: {expected_min} to {expected_max}", file=sys.stderr)
                        for outlier_ts in sorted(outliers):
                            outlier_dt = datetime.fromtimestamp(outlier_ts)
                            diff_from_min = outlier_ts - expected_min
                            diff_from_max = outlier_ts - expected_max
                            print(f"    Outlier: {outlier_dt} (Unix: {outlier_ts}) - {diff_from_min:+d}s from first, {diff_from_max:+d}s from last", file=sys.stderr)
                    
                    # Log timestamp distribution
                    unique_timestamps = sorted(set(int(ts / 1000) for ts in all_timestamps))
                    print(f"  Unique timestamps: {len(unique_timestamps)} (expected: {len(intervals)})", file=sys.stderr)
                    if len(unique_timestamps) != len(intervals):
                        print(f"  WARNING: Number of unique timestamps ({len(unique_timestamps)}) doesn't match number of intervals ({len(intervals)})", file=sys.stderr)
                    
                    # Check for duplicate/constant values after test completion
                    # Group samples by timestamp and check if values remain constant
                    timestamp_to_samples: Dict[int, List[Tuple[str, float]]] = {}
                    for ts in write_request.timeseries:
                        metric_name = None
                        for label in ts.labels:
                            if label.name == '__name__':
                                metric_name = label.value
                                break
                        for sample in ts.samples:
                            ts_s = int(sample.timestamp / 1000)
                            if ts_s not in timestamp_to_samples:
                                timestamp_to_samples[ts_s] = []
                            timestamp_to_samples[ts_s].append((metric_name or 'unknown', sample.value))
                    
                    # Check for timestamps after the last valid interval
                    timestamps_after_test = [ts for ts in unique_timestamps if ts > expected_max + 1]
                    if timestamps_after_test:
                        print(f"  WARNING: Found {len(timestamps_after_test)} timestamp(s) after test completion:", file=sys.stderr)
                        last_valid_samples = timestamp_to_samples.get(expected_max, [])
                        for ts_after in sorted(timestamps_after_test):
                            after_samples = timestamp_to_samples.get(ts_after, [])
                            after_dt = datetime.fromtimestamp(ts_after)
                            print(f"    Timestamp {after_dt} (Unix: {ts_after}): {len(after_samples)} samples", file=sys.stderr)
                            
                            # Check if values match the last valid interval (indicating duplicate data)
                            if last_valid_samples:
                                last_values = {name: val for name, val in last_valid_samples}
                                after_values = {name: val for name, val in after_samples}
                                matching = sum(1 for name in last_values if name in after_values and abs(last_values[name] - after_values[name]) < 0.0001)
                                if matching > len(after_values) * 0.8:  # 80% of values match
                                    print(f"      WARNING: {matching}/{len(after_values)} values match last valid interval - likely duplicate data!", file=sys.stderr)
                                    print(f"      This suggests the same data is being sent repeatedly after test completion.", file=sys.stderr)
                    
                    # Check for constant values across multiple consecutive timestamps (indicating duplicate sends)
                    # Group by metric name and check for constant values
                    metric_to_timestamps: Dict[str, List[Tuple[int, float]]] = {}
                    for ts in write_request.timeseries:
                        metric_name = None
                        for label in ts.labels:
                            if label.name == '__name__':
                                metric_name = label.value
                                break
                        if metric_name:
                            for sample in ts.samples:
                                ts_s = int(sample.timestamp / 1000)
                                if metric_name not in metric_to_timestamps:
                                    metric_to_timestamps[metric_name] = []
                                metric_to_timestamps[metric_name].append((ts_s, sample.value))
                    
                    # Check for metrics with constant values across multiple timestamps after test completion
                    constant_value_warnings = []
                    for metric_name, samples in metric_to_timestamps.items():
                        # Sort by timestamp
                        samples.sort(key=lambda x: x[0])
                        # Check for constant values in timestamps after test completion
                        constant_value = None
                        constant_count = 0
                        constant_start_ts = None
                        for ts_s, value in samples:
                            if ts_s > expected_max + 1:  # After test completion
                                if constant_value is None or abs(value - constant_value) < 0.0001:
                                    if constant_value is None:
                                        constant_value = value
                                        constant_start_ts = ts_s
                                    constant_count += 1
                                else:
                                    # Value changed, reset
                                    if constant_count >= 3:  # At least 3 consecutive constant values
                                        constant_value_warnings.append((metric_name, constant_start_ts, ts_s - 1, constant_value, constant_count))
                                    constant_value = value
                                    constant_start_ts = ts_s
                                    constant_count = 1
                        # Check if we ended with constant values
                        if constant_count >= 3 and constant_start_ts:
                            constant_value_warnings.append((metric_name, constant_start_ts, samples[-1][0], constant_value, constant_count))
                    
                    if constant_value_warnings:
                        print(f"  WARNING: Found {len(constant_value_warnings)} metric(s) with constant values after test completion:", file=sys.stderr)
                        for metric_name, start_ts, end_ts, value, count in constant_value_warnings:
                            start_dt = datetime.fromtimestamp(start_ts)
                            end_dt = datetime.fromtimestamp(end_ts)
                            print(f"    {metric_name}: constant value {value} from {start_dt} to {end_dt} ({count} timestamps)", file=sys.stderr)
                            print(f"      This indicates duplicate/constant data being sent after test completion.", file=sys.stderr)
            
            # Serialize and compress
            data = write_request.SerializeToString()
            
            # Save uncompressed data to debug file if specified (as JSON)
            if debug_file:
                try:
                    # Try new parameter name (protobuf 26.x+)
                    json_data = MessageToJson(write_request, always_print_fields_with_no_presence=True)  # type: ignore[call-arg]
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        f.write(json_data)
                    print(f"Saved uncompressed payload as JSON ({len(json_data)} bytes) to {debug_file}", file=sys.stderr)
                except TypeError:
                    # Fallback for protobuf versions that use the old parameter name
                    try:
                        json_data = MessageToJson(write_request, including_default_value_fields=True)  # type: ignore[call-arg]
                        with open(debug_file, 'w', encoding='utf-8') as f:
                            f.write(json_data)
                        print(f"Saved uncompressed payload as JSON ({len(json_data)} bytes) to {debug_file}", file=sys.stderr)
                    except TypeError:
                        # Fallback for older protobuf versions that don't support either parameter
                        try:
                            json_data = MessageToJson(write_request)
                            with open(debug_file, 'w', encoding='utf-8') as f:
                                f.write(json_data)
                            print(f"Saved uncompressed payload as JSON ({len(json_data)} bytes) to {debug_file}", file=sys.stderr)
                        except Exception as e:
                            print(f"Warning: Failed to write debug file {debug_file}: {e}", file=sys.stderr)
                    except Exception as e:
                        print(f"Warning: Failed to write debug file {debug_file}: {e}", file=sys.stderr)
                except Exception as e:
                    print(f"Warning: Failed to write debug file {debug_file}: {e}", file=sys.stderr)
            
            # If dry_run, skip sending
            if dry_run:
                print("Dry-run mode: Skipping actual send to endpoint", file=sys.stderr)
                return True
            
            compressed_data = snappy.compress(data)
            print(f"Sending {len(compressed_data)} bytes (uncompressed: {len(data)} bytes)", file=sys.stderr)
            
            # Send to remote write endpoint
            current_time = datetime.now()
            print(f"Sending metrics at {current_time.isoformat()}", file=sys.stderr)
            response = requests.post(
                self.remote_write_url,
                data=compressed_data,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200 or response.status_code == 204:
                print(f"Successfully sent metrics (status {response.status_code}) at {current_time.isoformat()}", file=sys.stderr)
                return True
            else:
                print(f"Error sending metrics: {response.status_code} - {response.text}", file=sys.stderr)
                print(f"Response headers: {dict(response.headers)}", file=sys.stderr)
                return False
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error: Could not connect to {self.remote_write_url}", file=sys.stderr)
            print("  Make sure Prometheus is running and the remote write receiver is enabled", file=sys.stderr)
            print("  Start Prometheus with: --web.enable-remote-write-receiver", file=sys.stderr)
            return False
        except Exception as e:
            print(f"Error in remote write: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return False
    
    def _collect_bucket_bounds(self, intervals: List[IntervalStats]) -> List[float]:
        """Collect and sort all unique bucket upper bounds from all intervals.
        
        Returns:
            List of float values representing bucket upper bounds, sorted in ascending order.
        """
        all_bucket_bounds = set()
        for stats in intervals:
            for bucket_range in stats.latency_buckets.keys():
                # Extract upper bound - bucket_range is now "float-float" format
                upper_str = bucket_range.split('-')[-1]
                try:
                    upper_float = float(upper_str)
                    all_bucket_bounds.add(upper_float)
                except ValueError:
                    # Fallback: try regex extraction if format is different
                    upper_match = re.search(BUCKET_UPPER_BOUND_PATTERN, upper_str)
                    if upper_match:
                        try:
                            upper_float = float(upper_match.group(1))
                            all_bucket_bounds.add(upper_float)
                        except ValueError:
                            continue
        
        return sorted(all_bucket_bounds)
    
    def _calculate_interval_latency_stats(self, stats: IntervalStats) -> Tuple[float, int]:
        """Calculate latency sum and count for an interval from bucket data.
        
        Returns:
            Tuple of (interval_latency_sum, interval_latency_count)
        """
        interval_latency_sum = 0.0
        interval_latency_count = 0
        
        for bucket_range, count in stats.latency_buckets.items():
            interval_latency_count += count
            bucket_parts = bucket_range.split('-')
            if len(bucket_parts) == 2:
                try:
                    lower = float(bucket_parts[0])
                    upper = float(bucket_parts[1])
                    midpoint = (lower + upper) / 2.0
                    interval_latency_sum += midpoint * count
                except ValueError:
                    continue
        
        return interval_latency_sum, interval_latency_count
    
    def _process_interval_basic_metrics(self, stats: IntervalStats, time_series_map: Dict[tuple, Any],
                                        cumulative_queries_sent: int, cumulative_queries_completed: int,
                                        cumulative_queries_lost: int, cumulative_response_codes: Dict[str, int],
                                        timestamp_ms: int):
        """Process basic metrics (queries, response codes, packet sizes, performance) for an interval."""
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
        
        # Latency averages
        self._add_sample_to_map(time_series_map, 'dnsperf_latency_seconds_avg', {}, stats.avg_latency, timestamp_ms)
        self._add_sample_to_map(time_series_map, 'dnsperf_latency_seconds_stddev', {}, stats.latency_stddev, timestamp_ms)
    
    def _process_interval_latency_buckets(self, stats: IntervalStats, time_series_map: Dict[tuple, Any],
                                         sorted_bucket_bounds: List[float], cumulative_bucket_counts: Dict[str, int],
                                         cumulative_latency_count: int, timestamp_ms: int) -> Dict[str, int]:
        """Process latency buckets for an interval, making them cumulative.
        
        Returns:
            Updated cumulative_bucket_counts dictionary
        """
        # Step 1: Collect per-bucket counts for this interval
        # Map from float upper bound to count
        # Use a helper to match float values with tolerance for floating point precision
        def find_matching_bound(upper_float: float, sorted_bounds: List[float]) -> Optional[float]:
            """Find the matching bound in sorted_bounds, handling floating point precision issues."""
            # First try exact match
            if upper_float in sorted_bounds:
                return upper_float
            # If not found, try to find the closest match (should be exact for same string input)
            # This handles cases where float conversion might introduce tiny differences
            for bound in sorted_bounds:
                if abs(bound - upper_float) < 1e-12:  # Very small tolerance for floating point errors
                    return bound
            return None
        
        interval_per_bucket_counts: Dict[float, int] = {}
        for bucket_range, count in stats.latency_buckets.items():
            # Extract upper bound - bucket_range preserves original string format
            upper_str = bucket_range.split('-')[-1]
            try:
                upper_float = float(upper_str)
                # Find the matching bound in sorted_bucket_bounds to ensure consistency
                matching_bound = find_matching_bound(upper_float, sorted_bucket_bounds)
                if matching_bound is not None:
                    interval_per_bucket_counts[matching_bound] = count
                else:
                    # If no match found, use the float value directly (shouldn't happen normally)
                    interval_per_bucket_counts[upper_float] = count
            except ValueError:
                # Fallback: try regex extraction if format is different
                upper_match = re.search(BUCKET_UPPER_BOUND_PATTERN, upper_str)
                if upper_match:
                    try:
                        upper_float = float(upper_match.group(1))
                        matching_bound = find_matching_bound(upper_float, sorted_bucket_bounds)
                        if matching_bound is not None:
                            interval_per_bucket_counts[matching_bound] = count
                        else:
                            interval_per_bucket_counts[upper_float] = count
                    except ValueError:
                        continue
        
        # Step 2: Convert per-bucket counts to cumulative counts within this interval
        # Map from float upper bound to cumulative count
        cumulative_interval_counts: Dict[float, int] = {}
        running_cumulative = 0
        for bound_float in sorted_bucket_bounds:
            running_cumulative += interval_per_bucket_counts.get(bound_float, 0)
            cumulative_interval_counts[bound_float] = running_cumulative
        
        # Step 3: Accumulate cumulative counts across intervals
        # Convert float bounds to string labels for storage
        for bound_float in sorted_bucket_bounds:
            bound_str = format_bound_for_label(bound_float)
            interval_cumulative = cumulative_interval_counts.get(bound_float, 0)
            if bound_str not in cumulative_bucket_counts:
                cumulative_bucket_counts[bound_str] = 0
            cumulative_bucket_counts[bound_str] += interval_cumulative
        
        # Step 4: Emit all buckets with cumulative counts
        for bound_float in sorted_bucket_bounds:
            bound_str = format_bound_for_label(bound_float)
            cumulative_count = cumulative_bucket_counts.get(bound_str, 0)
            self._add_sample_to_map(time_series_map, 'dnsperf_latency_bucket', {'le': bound_str}, cumulative_count, timestamp_ms)
        
        # Step 5: Add +Inf bucket (required for histogram_quantile)
        self._add_sample_to_map(time_series_map, 'dnsperf_latency_bucket', {'le': '+Inf'}, cumulative_latency_count, timestamp_ms)
        
        return cumulative_bucket_counts
    
    def _finalize_time_series(self, time_series_map: Dict[tuple, Any], write_request) -> None:
        """Add all time series to the write request."""
        for time_series in time_series_map.values():
            # Only add TimeSeries that have at least one sample
            if len(time_series.samples) > 0:
                new_ts = write_request.timeseries.add()
                new_ts.CopyFrom(time_series)
    
    def _print_metric_sample(self, time_series, timestamp_ms: int, value: float) -> None:
        """Print a single metric sample in verbose mode."""
        # Extract metric name and labels
        metric_name = None
        labels = {}
        for label in time_series.labels:
            if label.name == '__name__':
                metric_name = label.value
            else:
                labels[label.name] = label.value
        
        # Format labels
        if labels:
            label_str = ','.join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            metric_str = f'{metric_name}{{{label_str}}}'
        else:
            metric_str = metric_name
        
        # Convert timestamp to datetime
        timestamp_dt = datetime.fromtimestamp(timestamp_ms / 1000.0)
        
        # Print to stdout
        print(f"{timestamp_dt.isoformat()} {metric_str} {value}")
    
    def _convert_intervals_to_remote_write(self, intervals: List[IntervalStats], command_line_labels: Optional[Dict[str, str]] = None):
        """Convert interval statistics directly to remote write format."""
        write_request = prompb_pb2.WriteRequest()  # type: ignore
        time_series_map: Dict[tuple, Any] = {}
        
        if not intervals:
            return write_request
        
        # Add dnsperf_info metric with command line labels (only once, use first interval timestamp)
        # This metric should only be sent in the first timestamp
        if command_line_labels:
            first_timestamp_ms = int(intervals[0].timestamp.timestamp() * 1000)
            # Create labels dict with all command line parameters
            info_labels = command_line_labels.copy()
            # Only add dnsperf_info once with the first timestamp
            self._add_sample_to_map(time_series_map, 'dnsperf_info', info_labels, 1.0, first_timestamp_ms)
        
        # Determine the valid timestamp range based on intervals
        last_interval_ts = int(intervals[-1].timestamp.timestamp())
        valid_timestamp_max = last_interval_ts + 1  # Allow 1 second tolerance
        
        # Initialize cumulative counters
        cumulative_queries_sent = 0
        cumulative_queries_completed = 0
        cumulative_queries_lost = 0
        cumulative_response_codes: Dict[str, int] = {}
        cumulative_latency_sum = 0.0
        cumulative_latency_count = 0
        cumulative_bucket_counts: Dict[str, int] = {}
        
        # Collect all bucket bounds (already sorted as floats)
        sorted_bucket_bounds = self._collect_bucket_bounds(intervals)
        
        # Track timestamps for logging
        interval_timestamps = []
        skipped_intervals = []
        
        # Process each interval
        for stats in intervals:
            timestamp_ms = int(stats.timestamp.timestamp() * 1000)
            timestamp_s = int(stats.timestamp.timestamp())
            
            # Skip intervals with timestamps beyond the valid range (likely duplicates or errors)
            if timestamp_s > valid_timestamp_max:
                skipped_intervals.append((stats.interval_number, timestamp_s, stats.timestamp))
                continue
            
            interval_timestamps.append((stats.interval_number, timestamp_s, stats.run_time, stats.timestamp))
            
            # Log first, last, and any suspicious intervals
            if stats.interval_number == 1:
                print(f"  Interval {stats.interval_number}: timestamp={stats.timestamp} (Unix: {timestamp_s}), run_time={stats.run_time:.6f}s", file=sys.stderr)
            elif stats.interval_number == len(intervals):
                print(f"  Interval {stats.interval_number}: timestamp={stats.timestamp} (Unix: {timestamp_s}), run_time={stats.run_time:.6f}s", file=sys.stderr)
            elif stats.interval_number <= 5 or stats.interval_number >= len(intervals) - 4:
                # Log first 5 and last 5 intervals
                print(f"  Interval {stats.interval_number}: timestamp={stats.timestamp} (Unix: {timestamp_s}), run_time={stats.run_time:.6f}s", file=sys.stderr)
            
            # Accumulate counters
            cumulative_queries_sent += stats.queries_sent
            cumulative_queries_completed += stats.queries_completed
            cumulative_queries_lost += stats.queries_lost
            
            for code, count in stats.response_codes.items():
                if code not in cumulative_response_codes:
                    cumulative_response_codes[code] = 0
                cumulative_response_codes[code] += count
            
            # Process basic metrics
            self._process_interval_basic_metrics(
                stats, time_series_map, cumulative_queries_sent, cumulative_queries_completed,
                cumulative_queries_lost, cumulative_response_codes, timestamp_ms
            )
            
            # Process latency statistics
            interval_latency_sum, interval_latency_count = self._calculate_interval_latency_stats(stats)
            cumulative_latency_sum += interval_latency_sum
            cumulative_latency_count += interval_latency_count
            
            self._add_sample_to_map(time_series_map, 'dnsperf_latency_seconds_sum', {}, cumulative_latency_sum, timestamp_ms)
            self._add_sample_to_map(time_series_map, 'dnsperf_latency_seconds_count', {}, cumulative_latency_count, timestamp_ms)
            
            # Process latency buckets
            cumulative_bucket_counts = self._process_interval_latency_buckets(
                stats, time_series_map, sorted_bucket_bounds, cumulative_bucket_counts,
                cumulative_latency_count, timestamp_ms
            )
        
        # Finalize all time series
        self._finalize_time_series(time_series_map, write_request)
        
        # Log summary of timestamp distribution
        if skipped_intervals:
            print(f"  WARNING: Skipped {len(skipped_intervals)} interval(s) with timestamps beyond valid range:", file=sys.stderr)
            print(f"    Valid timestamp range: up to {valid_timestamp_max} (Unix: {valid_timestamp_max})", file=sys.stderr)
            for interval_num, ts_s, ts_dt in skipped_intervals:
                print(f"    Skipped interval {interval_num}: timestamp={ts_dt} (Unix: {ts_s}) - {ts_s - valid_timestamp_max}s beyond valid range", file=sys.stderr)
                print(f"      This likely indicates duplicate data being sent after test completion.", file=sys.stderr)
        
        if interval_timestamps:
            print(f"Processed {len(interval_timestamps)} intervals with timestamps", file=sys.stderr)
            # Check for timestamp anomalies
            prev_ts = None
            prev_run_time = None
            for interval_num, ts_s, run_time, ts_dt in interval_timestamps:
                if prev_ts is not None:
                    ts_diff = ts_s - prev_ts
                    run_time_diff = run_time - prev_run_time if prev_run_time is not None else run_time
                    # Warn if timestamp jump is significantly different from run_time difference
                    if abs(ts_diff - run_time_diff) > 2.0:  # 2 second tolerance
                        print(f"  WARNING: Interval {interval_num} has timestamp jump of {ts_diff}s but run_time difference of {run_time_diff:.6f}s", file=sys.stderr)
                        print(f"    Previous: interval={interval_num-1}, timestamp={prev_ts}, run_time={prev_run_time:.6f}s", file=sys.stderr)
                        print(f"    Current: interval={interval_num}, timestamp={ts_s}, run_time={run_time:.6f}s", file=sys.stderr)
                prev_ts = ts_s
                prev_run_time = run_time
        
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
            time_series = types_pb2.TimeSeries()  # type: ignore
            
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
        
        # For _info metrics, only add the first sample (first timestamp)
        # Skip if samples already exist
        if metric_name.endswith('_info') and len(time_series_map[key].samples) > 0:
            return
        
        # Add sample to the existing TimeSeries
        sample = time_series_map[key].samples.add()
        sample.value = value
        sample.timestamp = timestamp_ms
        
        # Print verbose output if enabled
        if self.verbose:
            self._print_metric_sample(time_series_map[key], timestamp_ms, value)

