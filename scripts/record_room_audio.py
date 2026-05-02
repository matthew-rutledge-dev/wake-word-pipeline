#!/usr/bin/env python3
"""Record ambient room audio from Voice PE satellites via ESPHome API.

Connects to an ESP32 Voice PE satellite, subscribes as the voice assistant
handler, and triggers microphone streaming via the start_recording_direct_audio
API service. Saves as 16 kHz mono WAV files suitable for openWakeWord training.

Usage:
    python scripts/record_room_audio.py --device masterai --duration 600
    python scripts/record_room_audio.py --device officeai --duration 1800 --split
    python scripts/record_room_audio.py --device all --duration 600
    python scripts/record_room_audio.py --list-devices

How it works:
  1. Script disables the HA config entry for the target device (WebSocket API)
  2. This disconnects HA's API client, freeing the VA subscription slot
  3. Script connects and subscribes as the voice assistant handler
  4. Script calls start_recording_direct_audio service to trigger mic streaming
  5. Audio streams continuously — script manages segment timing (default 30s)
  6. After each segment, RUN_END stops the pipeline, then next segment auto-starts
  7. Segments concatenate into one WAV (or split with --split)
  8. Ctrl+C saves what's been captured and exits
  9. Script re-enables the HA config entry — HA auto-reconnects

Requires: aioesphomeapi, websockets
"""

import argparse
import asyncio
import json
import os
import subprocess
import struct
import sys
import time
import wave
from pathlib import Path

from aioesphomeapi import APIClient, VoiceAssistantEventType

try:
    import websockets
except ImportError:
    websockets = None

# Voice PE satellites and their ESPHome API credentials
# Device name -> (IP, noise_psk, friendly_name, room)
DEVICES = {
    "masterai": (
        "192.168.88.93",
        "u4HROMSqA4wD7y+H7eDuE111BWcvxrmmtjk3ddipflY=",
        "Home Assistant Voice 092f16",
        "master_bedroom",
    ),
    "officeai": (
        "192.168.95.217",
        "DaaoXn+MyvcDU9gCM4IFxt/C25+0vA/ytvDT/dspYgA=",
        "Home Assistant Voice 091a7e",
        "office",
    ),
    "kitchenai": (
        "192.168.88.5",
        "CS1mpFXCWbRu8nU4C48apQSFqLhFyYAL61rwLB8vzHY=",
        "Home Assistant Voice 091ed3",
        "kitchen",
    ),
    "oldphone": (
        "192.168.91.3",
        "1fxEAu2rl/yIBS6pSSxjvGf80idx4Wk090rO4ECapQQ=",
        "Home Assistant Voice 0a2592",
        "hallway",
    ),
    "korin": (
        "192.168.86.212",
        "ZoGz7pNhdXa4T8lmL9owXFiejmo8gEcLu8dcPwYGw7w=",
        "ESP_ONE_32_M5_1 (Korin)",
        "living_room",
    ),
    "bond": (
        "192.168.86.214",
        "2bR0ft9tA2dPF8dpXU9IRgUDwz7pyOyanysuNIG6EZY=",
        "Bond",
        "bond_room",
    ),
    "ranga": (
        "192.168.86.211",
        "0em8PEg7XbqLqL6W47PsjouzCJ+FFO0rTWS3qEAVy98=",
        "Ranga",
        "ranga_room",
    ),
    "puar": (
        "192.168.86.210",
        "K7iXpXh7JEjuo1b28hIraEopF3nu7yYPSGZXVn08e3E=",
        "Puar",
        "puar_room",
    ),
}

# HA config entry IDs for each ESPHome device (used to disable/enable)
DEVICE_CONFIG_ENTRIES = {
    "masterai": "01JVTXXRB9P94ZV95MVPTS6TGV",
    "officeai": "01JK3RCPPT1AKF6EHASF4FBH5J",
    "kitchenai": "01JVW7EJYNWDTG3ZB26SR7S6TB",
    "oldphone": "01K3VNFWCMR0GQ9CMT1GB8F33J",
    "korin": "c9a0058c480d1d2e3b748348781cf588",
    "bond": "01KQ7FZ6BRA290JKZEBR4XTQTP",
    "ranga": "01KQ7PMCSGKWVDV5ND56TE4S4D",
    "puar": "876596d86988a4288edb135ff39b294a",
}

HA_WS_URL = "ws://192.168.86.29:8123/api/websocket"

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # 16-bit
CHANNELS = 1
BYTES_PER_SEC = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS

# Maximum segment duration in seconds (device pipeline can stream indefinitely)
SEGMENT_DURATION = 30.0

# Delay between segments to let the device settle before re-triggering
INTER_SEGMENT_DELAY = 2.0


def get_ha_token() -> str:
    """Get HA long-lived access token from env or infravault."""
    token = os.environ.get("HA_TOKEN")
    if token:
        return token

    try:
        result = subprocess.run(
            ["sudo", "-u", "infravault", "pwsh", "-c",
             ". /home/infravault/.local/share/infra/infra-secrets.ps1; "
             "Get-InfraSecretField 'HA_MCP_TOKEN' 'token'"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return ""


async def set_ha_entry_disabled(entry_id: str, disabled: bool, token: str) -> bool:
    """Disable or enable an HA config entry via WebSocket API."""
    if not websockets:
        print("ERROR: websockets module not installed (pip install websockets)")
        return False
    if not token:
        print("ERROR: No HA token available. Set HA_TOKEN env var or configure vault.")
        return False

    try:
        async with websockets.connect(HA_WS_URL) as ws:
            msg = json.loads(await ws.recv())
            if msg.get("type") != "auth_required":
                return False

            await ws.send(json.dumps({"type": "auth", "access_token": token}))
            msg = json.loads(await ws.recv())
            if msg.get("type") != "auth_ok":
                print(f"ERROR: HA auth failed: {msg}")
                return False

            await ws.send(json.dumps({
                "id": 1,
                "type": "config_entries/disable",
                "entry_id": entry_id,
                "disabled_by": "user" if disabled else None,
            }))
            msg = json.loads(await ws.recv())
            return msg.get("success", False)
    except Exception as e:
        print(f"ERROR: WebSocket call failed: {e}")
        return False


class RoomRecorder:
    """Records ambient audio from a Voice PE satellite."""

    def __init__(self, device_name: str, target_duration: float, output_path: Path,
                 split_segments: bool = False, ha_token: str = ""):
        if device_name not in DEVICES:
            raise ValueError(f"Unknown device '{device_name}'. Available: {', '.join(DEVICES)}")
        self.device_name = device_name
        self.ip, self.psk, self.friendly, self.room = DEVICES[device_name]
        self.target_duration = target_duration
        self.output_path = output_path
        self.split_segments = split_segments
        self.ha_token = ha_token
        self.config_entry_id = DEVICE_CONFIG_ENTRIES.get(device_name, "")

        self.audio_buffer = bytearray()
        self.segment_buffer = bytearray()
        self.segments: list[bytearray] = []
        self.session_active = False
        self.total_recorded = 0.0
        self.segment_count = 0
        self.stop_event = asyncio.Event()
        self.start_event = asyncio.Event()
        self.client: APIClient | None = None
        self._recording_service = None
        self._ha_was_disabled = False

    async def _find_recording_service(self):
        """Discover the start_recording_direct_audio service on the device."""
        _, services = await self.client.list_entities_services()
        for svc in services:
            if svc.name == "start_recording_direct_audio":
                self._recording_service = svc
                return True
        return False

    async def _trigger_recording(self):
        """Call the device's start_recording_direct_audio service."""
        if self._recording_service and self.client:
            await self.client.execute_service(self._recording_service, {})

    async def _stop_pipeline(self):
        """Send RUN_END to stop the current voice pipeline."""
        if self.client:
            self.client.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END, {})

    async def _disable_ha(self):
        """Disable the HA config entry to release the VA subscription."""
        if not self.config_entry_id:
            print(f"  WARNING: No config entry ID for {self.device_name}, skipping HA disable")
            return False
        if not self.ha_token:
            print("  WARNING: No HA token — cannot disable entry. Recording may fail.")
            return False

        print(f"  Disabling HA entry for {self.device_name}...")
        ok = await set_ha_entry_disabled(self.config_entry_id, True, self.ha_token)
        if ok:
            self._ha_was_disabled = True
            print(f"  HA disconnected from {self.device_name}")
            await asyncio.sleep(3)  # Wait for HA to release the connection
        else:
            print(f"  WARNING: Failed to disable HA entry — recording may fail")
        return ok

    async def _enable_ha(self):
        """Re-enable the HA config entry."""
        if not self._ha_was_disabled:
            return
        print(f"  Re-enabling HA entry for {self.device_name}...")
        ok = await set_ha_entry_disabled(self.config_entry_id, False, self.ha_token)
        if ok:
            self._ha_was_disabled = False
            print(f"  HA reconnecting to {self.device_name}")
        else:
            print(f"  WARNING: Failed to re-enable HA entry!")
            print(f"  Manual fix: HA UI > Settings > Devices > {self.friendly} > Enable")

    async def handle_start(self, conversation_id, flags, audio_settings, wake_word_phrase):
        """Called when device starts a voice pipeline (triggered by our service call)."""
        self.session_active = True
        self.segment_buffer = bytearray()
        self.start_event.set()

        # Send pipeline events to keep the device streaming
        if self.client:
            self.client.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_RUN_START, {})
            self.client.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_STT_START, {})

        return 0  # Use API audio transport

    async def handle_stop(self, abort):
        """Called when the device ends the voice session."""
        self.session_active = False

    async def handle_audio(self, data):
        """Called with raw audio data from the device microphone."""
        if self.session_active:
            self.segment_buffer.extend(data)

    async def _record_segment(self) -> bool:
        """Record one segment. Returns True if more recording is needed."""
        self.segment_count += 1
        remaining = self.target_duration - self.total_recorded
        seg_target = min(SEGMENT_DURATION, remaining)

        self.start_event.clear()
        self.session_active = False
        self.segment_buffer = bytearray()

        # Trigger recording
        await self._trigger_recording()

        # Wait for handle_start
        try:
            await asyncio.wait_for(self.start_event.wait(), timeout=10)
        except asyncio.TimeoutError:
            print(f"  [Segment {self.segment_count}] ERROR: Device did not start pipeline")
            return False

        print(f"  [Segment {self.segment_count}] RECORDING {seg_target:.0f}s...")

        # Collect audio for seg_target seconds
        seg_start = time.time()
        last_report = 0
        while time.time() - seg_start < seg_target:
            await asyncio.sleep(0.5)
            elapsed = time.time() - seg_start
            seg_secs = len(self.segment_buffer) / BYTES_PER_SEC
            total_secs = self.total_recorded + seg_secs
            if int(elapsed) >= last_report + 5:
                last_report = int(elapsed)
                sys.stdout.write(
                    f"\r    ... {seg_secs:.0f}s segment, {total_secs:.0f}s total   ")
                sys.stdout.flush()

        # Stop the pipeline
        self.session_active = False
        await self._stop_pipeline()
        await asyncio.sleep(0.2)  # Brief drain

        seg_secs = len(self.segment_buffer) / BYTES_PER_SEC
        if self.segment_buffer:
            self.segments.append(bytearray(self.segment_buffer))
            self.audio_buffer.extend(self.segment_buffer)

        self.total_recorded = len(self.audio_buffer) / BYTES_PER_SEC
        remaining = max(0, self.target_duration - self.total_recorded)

        print(f"\n  [Segment {self.segment_count}] Captured {seg_secs:.1f}s. "
              f"Total: {self.total_recorded:.1f}/{self.target_duration:.0f}s "
              f"({remaining:.0f}s remaining)")

        return remaining > 0

    def save_wav(self):
        """Save the accumulated audio buffer as WAV file(s)."""
        # Include any partial segment from an interrupted recording
        if self.session_active and self.segment_buffer:
            self.segments.append(bytearray(self.segment_buffer))
            self.audio_buffer.extend(self.segment_buffer)
            self.session_active = False

        if not self.audio_buffer:
            print("No audio to save.")
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.split_segments and len(self.segments) > 1:
            stem = self.output_path.stem
            suffix = self.output_path.suffix
            parent = self.output_path.parent
            for i, seg in enumerate(self.segments, 1):
                seg_path = parent / f"{stem}_seg{i:03d}{suffix}"
                self._write_wav(seg_path, seg)
                dur = len(seg) / BYTES_PER_SEC
                print(f"  Segment {i}: {seg_path.name} ({dur:.1f}s)")
            print(f"Saved {len(self.segments)} segment files in {parent}")
        else:
            self._write_wav(self.output_path, self.audio_buffer)

        duration = len(self.audio_buffer) / BYTES_PER_SEC
        size_mb = len(self.audio_buffer) / (1024 * 1024)
        print(f"\nTotal saved: {duration:.1f}s ({duration/60:.1f} min), {size_mb:.1f} MB")
        print(f"Format: {SAMPLE_RATE}Hz, {SAMPLE_WIDTH*8}-bit, mono WAV")

    @staticmethod
    def _write_wav(path: Path, data: bytearray):
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(bytes(data))

    async def record(self):
        """Main recording loop."""
        # Step 1: Disable HA's connection to this device
        await self._disable_ha()

        self.client = APIClient(
            address=self.ip, port=6053, password="", noise_psk=self.psk)

        print(f"Connecting to {self.device_name} ({self.ip})...")
        try:
            await self.client.connect(login=True)
        except Exception as e:
            print(f"Connection failed: {e}")
            await self._enable_ha()
            return False

        try:
            info = await self.client.device_info()
            print(f"Connected: {info.name} ({info.mac_address})")
            print(f"Room: {self.room}")
            print(f"Target: {self.target_duration:.0f}s ({self.target_duration/60:.1f} min)")
            print(f"Output: {self.output_path}")
            if self.split_segments:
                print("Mode: Split segments into individual WAVs")

            if not await self._find_recording_service():
                print("ERROR: start_recording_direct_audio service not found on device!")
                print("Flash firmware with the service first.")
                return False

            unsub = self.client.subscribe_voice_assistant(
                handle_start=self.handle_start,
                handle_stop=self.handle_stop,
                handle_audio=self.handle_audio,
            )
            print(f"\n--- Voice control PAUSED for {self.device_name} ---")
            print(f"Recording automatically — {SEGMENT_DURATION:.0f}s segments.")
            print(f"Ctrl+C to save and exit at any time.\n")

            await asyncio.sleep(1.0)  # Let subscription settle

            # Record segments until target duration reached
            try:
                while True:
                    more = await self._record_segment()
                    if not more:
                        break
                    print(f"  Next segment in {INTER_SEGMENT_DELAY:.0f}s...")
                    await asyncio.sleep(INTER_SEGMENT_DELAY)
            except asyncio.CancelledError:
                pass

            unsub()

        finally:
            try:
                await self.client.disconnect()
            except Exception:
                pass
            await self._enable_ha()
            print(f"\n--- Voice control RESUMED for {self.device_name} ---")

        self.save_wav()
        return True


async def record_all_rooms(target_duration: float, output_dir: Path, split: bool,
                           ha_token: str):
    """Record from all Voice PE satellites sequentially."""
    voice_pe_devices = [d for d in DEVICES if d != "korin"]
    print(f"Recording from {len(voice_pe_devices)} Voice PE satellites")
    print(f"Target: {target_duration:.0f}s per room, output: {output_dir}\n")

    for device_name in voice_pe_devices:
        _, _, _, room = DEVICES[device_name]
        output_path = output_dir / f"{room}_{device_name}.wav"
        recorder = RoomRecorder(device_name, target_duration, output_path, split,
                                ha_token=ha_token)
        print(f"\n{'='*60}")
        print(f"Room: {room} (device: {device_name})")
        print(f"{'='*60}")
        await recorder.record()
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Record ambient room audio from Voice PE satellites")
    parser.add_argument("--device", choices=list(DEVICES.keys()) + ["all"],
        help="Device to record from (or 'all' for sequential)")
    parser.add_argument("--duration", type=float, default=600,
        help="Target duration in seconds (default: 600 = 10 min)")
    parser.add_argument("--output", type=str,
        help="Output path (default: negative_audio/<room>_<device>.wav)")
    parser.add_argument("--split", action="store_true",
        help="Save each segment as a separate WAV file")
    parser.add_argument("--ha-token", type=str,
        help="HA long-lived access token (default: env HA_TOKEN or infravault)")
    parser.add_argument("--list-devices", action="store_true",
        help="List available devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        print("Available Voice PE satellites:")
        print(f"  {'Name':<12} {'IP':<18} {'Room':<16} {'Type'}")
        print(f"  {'-'*12} {'-'*18} {'-'*16} {'-'*20}")
        for name, (ip, _, friendly, room) in DEVICES.items():
            t = "Voice PE" if name != "korin" else "ESP32 (no PSRAM)"
            print(f"  {name:<12} {ip:<18} {room:<16} {t}")
        return

    if not args.device:
        parser.error("--device is required (or use --list-devices)")

    # Get HA token for config entry management
    ha_token = args.ha_token or get_ha_token()
    if not ha_token:
        print("WARNING: No HA token available. HA will not be disconnected during recording.")
        print("  Set HA_TOKEN env var, pass --ha-token, or configure infravault.")
        print("  Recording will likely fail (HA holds the VA subscription).")
        resp = input("  Continue anyway? [y/N] ")
        if resp.lower() != "y":
            return

    base_dir = Path(__file__).resolve().parent.parent / "negative_audio"

    if args.device == "all":
        output_dir = Path(args.output) if args.output else base_dir
        try:
            asyncio.run(record_all_rooms(args.duration, output_dir, args.split,
                                         ha_token))
        except KeyboardInterrupt:
            print("\nRecording interrupted.")
    else:
        _, _, _, room = DEVICES[args.device]
        default_out = base_dir / f"{room}_{args.device}.wav"
        output_path = Path(args.output) if args.output else default_out
        recorder = RoomRecorder(args.device, args.duration, output_path, args.split,
                                ha_token=ha_token)
        try:
            asyncio.run(recorder.record())
        except KeyboardInterrupt:
            print("\nRecording interrupted — saving captured audio...")
            recorder.save_wav()


if __name__ == "__main__":
    main()
