#!/usr/bin/env python3
"""
Parse dnsperf statistics and convert them to Prometheus metrics,
then send them using Prometheus remote write.
"""

import sys
import os
import argparse
from zoneinfo import ZoneInfo

# Handle both relative imports (when used as module) and absolute imports (when run as script)
try:
    from .parser import DnsperfParser
    from .utils import prepare_headers, send_metrics_remote_write
except ImportError:
    # If relative imports fail, we're running as a script - add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from parser.parser import DnsperfParser
    from parser.utils import prepare_headers, send_metrics_remote_write


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
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print timestamp and metric information to stdout for each metric sent to Prometheus'
    )
    parser.add_argument(
        '--debug-file',
        help='Save the uncompressed payload data (before snappy compression) as JSON to the specified file for debugging'
    )
    parser.add_argument(
        '--timezone',
        default='UTC',
        help='Timezone of timestamps in the dnsperf file (default: UTC). Use IANA timezone names like "Europe/Madrid", "America/New_York", "UTC"'
    )
    
    args = parser.parse_args()
    
    # Validate timezone
    try:
        tz = ZoneInfo(args.timezone)
    except Exception as e:
        print(f"Error: Invalid timezone '{args.timezone}': {e}", file=sys.stderr)
        sys.exit(1)
    
    # Parse dnsperf stats
    print(f"Parsing dnsperf statistics from {args.input_file}...")
    print(f"Using timezone: {args.timezone}")
    dnsperf_parser = DnsperfParser(args.input_file, timezone=tz)
    intervals = dnsperf_parser.parse()
    
    if not intervals:
        print("No interval statistics found in file.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(intervals)} interval(s)")
    
    headers = prepare_headers(args.remote_write_header)
    send_metrics_remote_write(
        args.remote_write_url, headers, intervals, args.instance_label, args.verbose, args.dry_run, args.debug_file,
        command_line_labels=dnsperf_parser.command_line_labels
    )


if __name__ == '__main__':
    main()
