"""Parser for dnsperf statistics output."""

import re
import sys
from typing import List, Optional, Dict, Union
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from .models import IntervalStats
from .utils import BUCKET_UPPER_BOUND_PATTERN


class DnsperfParser:
    """Parser for dnsperf statistics output."""
    
    def __init__(self, file_path: str, timezone: Union[ZoneInfo, timezone] = timezone.utc):
        """Initialize the parser.
        
        Args:
            file_path: Path to the dnsperf statistics file
            timezone: Timezone of timestamps in the dnsperf file (default: UTC)
        """
        self.file_path = file_path
        self.timezone = timezone
        self.intervals: List[IntervalStats] = []
        self.start_time: Optional[datetime] = None
        self.command_line_labels: Dict[str, str] = {}
        
    def parse(self) -> List[IntervalStats]:
        """Parse the dnsperf stats file and return list of interval statistics."""
        with open(self.file_path, 'r') as f:
            content = f.read()
        
        # Parse command line to extract parameters as labels
        self.command_line_labels = self._parse_command_line(content)
        
        # Parse the start timestamp and apply the configured timezone
        start_time_match = re.search(r'\[Status\] Started at:\s+(.+)', content)
        if start_time_match:
            start_time_str = start_time_match.group(1).strip()
            # Parse format like "Thu Nov 27 15:58:14 2025"
            try:
                # Try common date formats
                parsed_time = None
                for fmt in ['%a %b %d %H:%M:%S %Y', '%a %b %d %H:%M:%S %Y %Z']:
                    try:
                        parsed_time = datetime.strptime(start_time_str, fmt)
                        break
                    except ValueError:
                        continue
                if parsed_time is not None:
                    # Apply the configured timezone to the parsed time
                    self.start_time = parsed_time.replace(tzinfo=self.timezone)
                else:
                    print(f"Warning: Could not parse start time '{start_time_str}', using current time", file=sys.stderr)
                    self.start_time = datetime.now(self.timezone)
            except Exception as e:
                print(f"Warning: Error parsing start time: {e}, using current time", file=sys.stderr)
                self.start_time = datetime.now(self.timezone)
        else:
            print("Warning: Could not find start time, using current time", file=sys.stderr)
            self.start_time = datetime.now(self.timezone)
        
        # Stop parsing at "[Status] Testing complete" to avoid the final summary section
        # Split content at the testing complete marker
        testing_complete_match = re.search(r'\[Status\] Testing complete', content)
        if testing_complete_match:
            # Only parse content before the testing complete marker
            content = content[:testing_complete_match.start()]
        
        # Find all interval statistics sections
        # Use negative lookahead to match content until next marker
        interval_pattern = r'Interval Statistics:\s*\n\s*\n((?:(?!\n(?:Interval Statistics:|\[Status\]|Statistics:|\Z)).)+)'
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
            
            # Extract latency buckets (supports scientific notation)
            latency_buckets = {}
            bucket_pattern = r'([\d.eE+-]+)\s+-\s+([\d.eE+-]+):\s+(\d+)'
            for bucket_match in re.finditer(bucket_pattern, text):
                lower_str = bucket_match.group(1)
                upper_str = bucket_match.group(2)
                count = int(bucket_match.group(3))
                # Store original string representation to preserve exact precision
                # Convert to float only for validation, but keep original strings
                try:
                    # Validate that both values can be converted to float
                    float(lower_str)  # Validate but don't store
                    float(upper_str)  # Validate but don't store
                    # Store using original string representation to preserve exact format
                    # This ensures 0.000044 stays as 0.000044, not 4.4e-05
                    bucket_range = f"{lower_str}-{upper_str}"
                    latency_buckets[bucket_range] = count
                except ValueError:
                    # If conversion fails, use original strings
                    bucket_range = f"{lower_str}-{upper_str}"
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
    
    def _parse_command_line(self, content: str) -> Dict[str, str]:
        """Parse the command line from dnsperf output and extract all parameters as labels.
        
        Returns:
            Dictionary of label names to values extracted from command line
        """
        labels: Dict[str, str] = {}
        
        # Extract command line
        cmd_line_match = re.search(r'\[Status\] Command line:\s+(.+)', content)
        if not cmd_line_match:
            return labels
        
        cmd_line = cmd_line_match.group(1).strip()
        
        # Map of short options to label names
        # Based on dnsperf -h output
        option_map = {
            '-f': 'family',
            '-m': 'mode',
            '-s': 'server',
            '-p': 'port',
            '-a': 'local_addr',
            '-x': 'local_port',
            '-d': 'datafile',
            '-c': 'clients',
            '-T': 'threads',
            '-n': 'maxruns',
            '-l': 'timelimit',
            '-b': 'buffer_size',
            '-t': 'timeout',
            '-y': 'tsig',
            '-q': 'max_queries_outstanding',
            '-Q': 'max_qps',
            '-S': 'stats_interval',
            '-E': 'edns_option',
        }
        
        # Boolean flags (no value)
        boolean_flags = {
            '-e': 'edns_enabled',
            '-D': 'dnssec_ok',
            '-u': 'dynamic_updates',
            '-B': 'tcp_stream_binary',
            '-v': 'verbose',
            '-W': 'log_warnings_to_stdout',
        }
        
        # Tokenize command line - split on whitespace but preserve quoted strings
        tokens = []
        current_token = ''
        in_quotes = False
        quote_char = None
        
        for char in cmd_line:
            if char in ('"', "'") and (not in_quotes or char == quote_char):
                if in_quotes and char == quote_char:
                    in_quotes = False
                    quote_char = None
                else:
                    in_quotes = True
                    quote_char = char
                current_token += char
            elif char.isspace() and not in_quotes:
                if current_token:
                    tokens.append(current_token)
                    current_token = ''
            else:
                current_token += char
        
        if current_token:
            tokens.append(current_token)
        
        # Parse tokens sequentially
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # Check for short options with values
            if token in option_map:
                label_name = option_map[token]
                if i + 1 < len(tokens):
                    value = tokens[i + 1]
                    # Remove quotes if present
                    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    labels[label_name] = value
                    i += 2
                else:
                    i += 1
            # Check for boolean flags
            elif token in boolean_flags:
                labels[boolean_flags[token]] = 'true'
                i += 1
            # Check for long options: -O name=value or -O name
            elif token == '-O':
                if i + 1 < len(tokens):
                    opt_value = tokens[i + 1]
                    # Check if it's name=value format
                    if '=' in opt_value:
                        option_name, option_value = opt_value.split('=', 1)
                    else:
                        option_name = opt_value
                        option_value = 'true'
                    
                    # Map long option names to label-friendly names
                    label_name = 'opt_' + option_name.replace('-', '_')
                    labels[label_name] = option_value
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        
        return labels

