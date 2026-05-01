#!/usr/bin/env python3
"""Record ambient room audio from Voice PE satellites via ESPHome API.

Connects to an ESP32 Voice PE satellite, subscribes as the voice assistant
handler, and captures microphone audio when triggered by the center button.
Saves as 16 kHz mono WAV files suitable for openWakeWord training negatives.

Usage:
    python scripts/record_room_audio.py --device masterai --duration 600
    python scripts/record_room_audio.py --device officeai --duration 1800 --split
    python scripts/record_room_audio.py --device all --duration 600
    python scripts/record_room_audio.py --list-devices

How it works:
  1. Script connects to the Voice PE and takes over as voice assistant handler
  2. HA voice control is PAUSED for this device during recording
  3. Press CENTER BUTTON on the device to start a recording segment
  4. Audio streams until the pipeline times out (~30s) or button pressed again
  5. Script auto-waits for the next button press to capture more segments
  6. Segments concatenate into one WAV (or split with --split)
  7. Ctrl+C saves what's been captured and exits
  8. After exit, HA automatically reconnects and resumes voice control

Requires: aioesphomeapi (pip install aioesphomeapi)
"""

import argparse
import asyncio
import struct
import sys
import time
import wave
from pathlib import Path

from aioesphomeapi import APIClient, VoiceAssistantEventType

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
}

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # 16-bit
CHANNELS = 1
BYTES_PER_SEC = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS


class RoomRecorder:
    """Records ambient audio from a Voice PE satellite."""

    def __init__(self, device_name: str, target_duration: float, output_path: Path,
                 split_segments: bool = False):
        if device_name not in DEVICES:
            raise ValueError(f"Unknown device '{device_name}'. Available: {', '.join(DEVICES)}")
        self.device_name = device_name
        self.ip, self.psk, self.friendly, self.room = DEVICES[device_name]
        self.target_duration = target_duration
        self.output_path = output_path
        self.split_segments = split_segments

        self.audio_buffer = bytearray()
        self.segment_buffer = bytearray()
        self.segments: list[bytearray] = []
        self.session_active = False
        self.session_start_time = 0.0
        self.total_recorded = 0.0
        self.segment_count = 0
        self.stop_event = asyncio.Event()
        self.client: APIClient | None = None

        self.device_sample_rate = SAMPLE_RATE
        self.device_channels = CHANNELS
        self.device_bits = 16

    async def handle_start(self, conversation_id, flags, audio_settings, wake_word_phrase):
        """Called when device requests a voice pipeline start (button press)."""
        self.device_sample_rate = audio_settings.raw_sample_rate
        self.device_channels = audio_settings.raw_channels
        self.device_bits = audio_settings.raw_bits_per_sample

        self.segment_count += 1
        self.session_active = True
        self.session_start_time = time.time()
        self.segment_buffer = bytearray()

        print(f"\n  [Segment {self.segment_count}] RECORDING "
              f"({self.device_sample_rate}Hz {self.device_channels}ch {self.device_bits}bit)")

        # Send pipeline events to keep the device streaming as long as possible
        if self.client:
            self.client.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_RUN_START, {})
            self.client.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_STT_START, {})

        return 0  # Use API audio transport

    async def handle_stop(self, abort):
        """Called when the device ends the voice session."""
        if not self.session_active:
            return
        self.session_active = False
        elapsed = time.time() - self.session_start_time
        seg_secs = len(self.segment_buffer) / BYTES_PER_SEC

        # Save segment
        if self.segment_buffer:
            self.segments.append(bytearray(self.segment_buffer))
            self.audio_buffer.extend(self.segment_buffer)

        self.total_recorded = len(self.audio_buffer) / BYTES_PER_SEC
        remaining = max(0, self.target_duration - self.total_recorded)

        print(f"  [Segment {self.segment_count}] Captured {seg_secs:.1f}s "
              f"(wall {elapsed:.1f}s). "
              f"Total: {self.total_recorded:.1f}/{self.target_duration:.0f}s")

        if self.total_recorded >= self.target_duration:
            print("  Target duration reached!")
            self.stop_event.set()
        else:
            print(f"  >>> Need {remaining:.0f}s more. Press CENTER BUTTON again.")

    async def handle_audio(self, data):
        """Called with raw audio data from the device microphone."""
        if not self.session_active:
            return

        audio_data = self._normalize_audio(data)
        self.segment_buffer.extend(audio_data)

        # Progress every 5 seconds
        seg_secs = len(self.segment_buffer) / BYTES_PER_SEC
        total_secs = (len(self.audio_buffer) + len(self.segment_buffer)) / BYTES_PER_SEC
        if int(seg_secs) > 0 and int(seg_secs) % 5 == 0:
            expected = int(seg_secs) * BYTES_PER_SEC
            if abs(len(self.segment_buffer) - expected) < BYTES_PER_SEC:
                sys.stdout.write(
                    f"\r    ... {seg_secs:.0f}s segment, {total_secs:.0f}s total   ")
                sys.stdout.flush()

    def _normalize_audio(self, data: bytes) -> bytes:
        """Convert raw audio to 16kHz 16-bit mono."""
        if self.device_bits == 16 and self.device_channels == 1:
            return data
        if self.device_bits == 32 and self.device_channels == 1:
            samples = struct.unpack(f"<{len(data)//4}i", data)
            return struct.pack(f"<{len(samples)}h", *(s >> 16 for s in samples))
        if self.device_channels == 2 and self.device_bits == 16:
            samples = struct.unpack(f"<{len(data)//2}h", data)
            mono = [(samples[i] + samples[i+1]) // 2 for i in range(0, len(samples), 2)]
            return struct.pack(f"<{len(mono)}h", *mono)
        if self.device_channels == 2 and self.device_bits == 32:
            samples = struct.unpack(f"<{len(data)//4}i", data)
            mono = [((samples[i]>>16) + (samples[i+1]>>16)) // 2 for i in range(0, len(samples), 2)]
            return struct.pack(f"<{len(mono)}h", *mono)
        return data

    def save_wav(self):
        """Save the accumulated audio buffer as WAV file(s)."""
        if not self.audio_buffer and not self.segments:
            # Save partial segment if we were interrupted mid-recording
            if self.segment_buffer:
                self.segments.append(bytearray(self.segment_buffer))
                self.audio_buffer.extend(self.segment_buffer)
            else:
                print("No audio to save.")
                return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.split_segments and len(self.segments) > 1:
            # Save each segment as a separate file
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
            # Single concatenated file
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
        self.client = APIClient(
            address=self.ip, port=6053, password="", noise_psk=self.psk)

        print(f"Connecting to {self.device_name} ({self.ip})...")
        try:
            await self.client.connect(login=True)
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

        info = await self.client.device_info()
        print(f"Connected: {info.name} ({info.mac_address})")
        print(f"Room: {self.room}")
        print(f"Target: {self.target_duration:.0f}s ({self.target_duration/60:.1f} min)")
        print(f"Output: {self.output_path}")
        if self.split_segments:
            print(f"Mode: Split segments into individual WAVs")

        unsub = self.client.subscribe_voice_assistant(
            handle_start=self.handle_start,
            handle_stop=self.handle_stop,
            handle_audio=self.handle_audio,
        )
        print(f"\n--- Voice control PAUSED for {self.device_name} ---")
        print(f"\n>>> Press CENTER BUTTON on the device to start recording")
        print(f">>> Each press captures one segment (~30s)")
        print(f">>> Repeat until target duration reached")
        print(f">>> Ctrl+C to save and exit at any time\n")

        try:
            while not self.stop_event.is_set():
                try:
                    await asyncio.wait_for(self.stop_event.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            pass
        finally:
            unsub()
            try:
                await self.client.disconnect()
            except Exception:
                pass
            print(f"\n--- Voice control RESUMED for {self.device_name} ---")

        self.save_wav()
        return True


async def record_all_rooms(target_duration: float, output_dir: Path, split: bool):
    """Record from all Voice PE satellites sequentially."""
    voice_pe_devices = [d for d in DEVICES if d != "korin"]
    print(f"Recording from {len(voice_pe_devices)} Voice PE satellites")
    print(f"Target: {target_duration:.0f}s per room, output: {output_dir}\n")

    for device_name in voice_pe_devices:
        _, _, _, room = DEVICES[device_name]
        output_path = output_dir / f"{room}_{device_name}.wav"
        recorder = RoomRecorder(device_name, target_duration, output_path, split)
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

    base_dir = Path(__file__).resolve().parent.parent / "negative_audio"

    if args.device == "all":
        output_dir = Path(args.output) if args.output else base_dir
        try:
            asyncio.run(record_all_rooms(args.duration, output_dir, args.split))
        except KeyboardInterrupt:
            print("\nRecording interrupted.")
    else:
        _, _, _, room = DEVICES[args.device]
        default_out = base_dir / f"{room}_{args.device}.wav"
        output_path = Path(args.output) if args.output else default_out
        recorder = RoomRecorder(args.device, args.duration, output_path, args.split)
        try:
            asyncio.run(recorder.record())
        except KeyboardInterrupt:
            print("\nRecording interrupted — saving captured audio...")
            recorder.save_wav()


if __name__ == "__main__":
    main()
