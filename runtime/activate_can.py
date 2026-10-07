#!/usr/bin/env python3
"""Configure the Piper SocketCAN interface without enabling or moving the arm."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.console import section, status

BITRATE = 1_000_000


def interface_name(value: str) -> str:
    """Validate a Linux interface name supplied as a positional or source argument."""
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,14}", value):
        raise argparse.ArgumentTypeError("Use 1-15 letters, digits, underscores, dots or hyphens")
    return value


def run_ip(ip: str, *arguments: str) -> str:
    """Run iproute2 with separate arguments and return stdout or a readable error."""
    result = subprocess.run([ip, *arguments], capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"ip failed with exit code {result.returncode}")
    return result.stdout


def activate_can(name: str = "can0", source: str | None = None) -> dict:
    """Select, optionally rename, and bring up one physical CAN interface.

    Args:
        name: Desired Linux interface name, default can0.
        source: Existing CAN interface to rename. Auto-selected when unambiguous.

    Returns:
        Verified iproute2 JSON record for the active 1 Mbps interface.
        No Piper SDK, motor-enable, or joint commands are used.
    """
    ip = shutil.which("ip")
    if ip is None:
        raise RuntimeError("ip not found; install iproute2 (iproute on openEuler)")
    links = json.loads(run_ip(ip, "-json", "-details", "link", "show"))
    by_name = {link["ifname"]: link for link in links}
    can_links = {key: link for key, link in by_name.items()
                 if link.get("linkinfo", {}).get("info_kind") == "can"}
    if not can_links:
        raise RuntimeError("No CAN interface found; check the USB-CAN connection")
    if name in by_name and name not in can_links:
        raise RuntimeError(f"Interface name {name} is occupied by a non-CAN device")
    if source:
        if source not in can_links:
            raise RuntimeError(f"Source CAN interface {source} not found; available: {', '.join(can_links)}")
        if name != source and name in by_name:
            raise RuntimeError(f"Target name {name} is occupied; cannot rename {source}")
    elif name in can_links:
        source = name
    elif len(can_links) == 1:
        source = next(iter(can_links))
    else:
        raise RuntimeError(f"Multiple CAN interfaces: {', '.join(can_links)}; select one with --source")

    link = can_links[source]
    data = link["linkinfo"].get("info_data", {})
    bitrate = data.get("bittiming", {}).get("bitrate")
    ready = (source == name and "UP" in link.get("flags", [])
             and bitrate == BITRATE and data.get("state") != "BUS-OFF")
    status("Interface", source if source == name else f"{source} -> {name}")
    if ready:
        status("Config", "Already UP; keeping the existing connection", "ok")
    else:
        if os.geteuid() != 0:
            raise RuntimeError("CAN configuration requires root; run as root or use sudo")
        if "UP" in link.get("flags", []):
            run_ip(ip, "link", "set", "dev", source, "down")
        if source != name:
            run_ip(ip, "link", "set", "dev", source, "name", name)
        run_ip(ip, "link", "set", "dev", name, "type", "can", "bitrate", str(BITRATE))
        run_ip(ip, "link", "set", "dev", name, "up")

    active = json.loads(run_ip(ip, "-json", "-details", "link", "show", "dev", name))[0]
    info = active.get("linkinfo", {}).get("info_data", {})
    if "UP" not in active.get("flags", []) or info.get("bittiming", {}).get("bitrate") != BITRATE:
        raise RuntimeError(f"{name} did not come UP at 1 Mbps")
    if info.get("state") == "BUS-OFF":
        raise RuntimeError(f"{name} is BUS-OFF; check arm power and CAN wiring")
    return active


def main() -> None:
    """Parse the optional alias and print the verified CAN activation result."""
    parser = argparse.ArgumentParser(
        description="Activate Piper CAN at 1 Mbps; no motor enable or motion commands.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("name", nargs="?", default="can0", type=interface_name,
                        help="Interface name/alias; automatically renames a unique CAN interface")
    parser.add_argument("--source", type=interface_name,
                        help="Original interface to rename when multiple CAN devices are connected")
    args = parser.parse_args()
    section("Piper CAN activation")
    try:
        active = activate_can(args.name, args.source)
    except (RuntimeError, OSError) as error:
        status("Error", str(error), "error")
        raise SystemExit(1) from None
    info = active["linkinfo"]["info_data"]
    status("CAN", f"{args.name} is UP", "ok")
    status("Bitrate", "1000000 bit/s (1 Mbps)")
    status("Bus state", info.get("state", "UNKNOWN"))
    status("Arm", "No motor enable or motion requested")
    if args.name != "can0":
        status("Inference", f"Add --can {args.name} when starting inference")


if __name__ == "__main__":
    main()
