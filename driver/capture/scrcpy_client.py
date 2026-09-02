"""
Real-Time Scrcpy H.264 TCP Stream Client.
Connects directly to Scrcpy daemon video socket, demuxes packets,
and pipes frames to H264Decoder and FrameBufferQueue.
"""
from __future__ import annotations

import os
import sys
import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import subprocess
import cv2
import numpy as np

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

from driver.capture.h264_decoder import FrameBufferQueue, H264Decoder
from driver.dispatcher.touch_dispatcher import find_adb_binary

logger = logging.getLogger(__name__)


@dataclass
class ScrcpyStreamMetadata:
    """Device metadata parsed from Scrcpy handshake."""
    device_name: str = ""
    codec_id: str = "h264"
    width: int = 1800
    height: int = 2880
    connected_at: float = 0.0


class ScrcpyStreamClient:
    """
    High-performance, low-latency Scrcpy TCP video stream client with
    automatic fallback to desktop window capture (MSS) and ADB screencap.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 27183,
        queue_size: int = 2,
        connect_timeout: float = 2.0,
        device_serial: Optional[str] = None,
        window_title: str = "scrcpy",
    ):
        self.host = host
        self.port = port
        self.device_serial = device_serial
        self.window_title = window_title
        self.connect_timeout = connect_timeout
        self.decoder = H264Decoder(output_format="bgr24")
        self.frame_queue = FrameBufferQueue(max_size=queue_size)
        self.metadata = ScrcpyStreamMetadata()

        self._socket: Optional[socket.socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._connected = False
        self._mode = "TCP"  # "TCP", "MSS", or "ADB"
        self._sct = (mss.MSS() if hasattr(mss, "MSS") else mss.mss()) if MSS_AVAILABLE else None

    @property
    def is_connected(self) -> bool:
        return self._connected and self._running

    def _find_window_geometry(self) -> Optional[Dict[str, int]]:
        """Locates scrcpy / emulator window geometry on Windows desktop."""
        if sys.platform != "win32":
            return None
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            found_hwnd = []

            def enum_proc(hwnd, lparam):
                if user32.IsWindowVisible(hwnd):
                    length = user32.GetWindowTextLengthW(hwnd)
                    if length > 0:
                        buff = ctypes.create_unicode_buffer(length + 1)
                        user32.GetWindowTextW(hwnd, buff, length + 1)
                        if self.window_title.lower() in buff.value.lower():
                            found_hwnd.append(hwnd)
                return True

            enum_type = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
            user32.EnumWindows(enum_type(enum_proc), 0)

            if found_hwnd:
                rect = wintypes.RECT()
                user32.GetWindowRect(found_hwnd[0], ctypes.byref(rect))
                return {
                    "top": rect.top,
                    "left": rect.left,
                    "width": rect.right - rect.left,
                    "height": rect.bottom - rect.top,
                }
        except Exception:
            pass
        return None

    def start(self) -> bool:
        """Connects and starts background frame reader thread."""
        with self._lock:
            if self._running:
                return True
            self._running = True

        # 1. Attempt TCP socket connection to Scrcpy daemon
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.connect_timeout)
            sock.connect((self.host, self.port))
            self._socket = sock
            self._connected = True
            self._mode = "TCP"
            self._thread = threading.Thread(target=self._tcp_read_loop, daemon=True)
            self._thread.start()
            logger.info("Scrcpy stream connected on TCP port %d", self.port)
            return True
        except Exception as e:
            logger.info("Scrcpy TCP socket connect skipped (%s). Initializing high-speed frame capture.", e)

        # 2. Check for Desktop Window Mirror (MSS)
        geom = self._find_window_geometry()
        if geom and self._sct:
            self._connected = True
            self._mode = "MSS"
            self._thread = threading.Thread(target=self._mss_capture_loop, args=(geom,), daemon=True)
            self._thread.start()
            logger.info("Scrcpy connected via Desktop Window Grabber (%s)", geom)
            return True

        # 3. ADB Frame Capture mode
        self._connected = True
        self._mode = "ADB"
        self._thread = threading.Thread(target=self._adb_capture_loop, daemon=True)
        self._thread.start()
        logger.info("Scrcpy fallback initialized via ADB screen pipe.")
        return True

    def _tcp_read_loop(self) -> None:
        """Demuxes raw H.264 packets from Scrcpy socket."""
        try:
            if not self._socket:
                return
            # Read handshake header (69 bytes dummy + dev_name + codec)
            header = self._socket.recv(69)
            if len(header) >= 69:
                self.metadata.device_name = header[1:65].decode("utf-8", errors="ignore").rstrip("\x00")
                self.metadata.codec_id = header[65:69].decode("utf-8", errors="ignore")

            while self._running and self._socket:
                # Scrcpy packet header: 8 bytes PTS + 4 bytes size
                pkt_hdr = self._recv_exact(12)
                if not pkt_hdr:
                    break
                pts, size = struct.unpack(">QI", pkt_hdr)
                payload = self._recv_exact(size)
                if not payload:
                    break

                frames = self.decoder.decode_chunk(payload)
                for f in frames:
                    self.frame_queue.put(f, timestamp=time.time())

        except Exception as e:
            logger.debug("Scrcpy TCP stream read terminated: %s", e)
        finally:
            self._connected = False

    def _recv_exact(self, num_bytes: int) -> Optional[bytes]:
        buf = bytearray()
        while len(buf) < num_bytes and self._running:
            try:
                chunk = self._socket.recv(num_bytes - len(buf)) if self._socket else None
                if not chunk:
                    return None
                buf.extend(chunk)
            except socket.timeout:
                continue
            except Exception:
                return None
        return bytes(buf)

    def _mss_capture_loop(self, geom: Dict[str, int]) -> None:
        """High-FPS window capture loop."""
        while self._running:
            try:
                if self._sct is None:
                    break
                sct_img = self._sct.grab(geom)
                img = np.array(sct_img)
                bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                if bgr.shape[:2] != (2880, 1800):
                    bgr = cv2.resize(bgr, (1800, 2880), interpolation=cv2.INTER_AREA)
                self.frame_queue.put(bgr, timestamp=time.time())
                time.sleep(0.016)  # ~60 fps
            except Exception:
                time.sleep(0.05)

    def _adb_capture_loop(self) -> None:
        """ADB screencap loop."""
        adb_bin = find_adb_binary()
        cmd = [adb_bin]
        if self.device_serial:
            cmd.extend(["-s", self.device_serial])
        cmd.extend(["exec-out", "screencap", "-p"])

        while self._running:
            t0 = time.perf_counter()
            try:
                res = subprocess.run(cmd, capture_output=True, timeout=2.0)
                if res.returncode == 0 and res.stdout:
                    arr = np.frombuffer(res.stdout, np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        if frame.shape[:2] != (2880, 1800):
                            frame = cv2.resize(frame, (1800, 2880), interpolation=cv2.INTER_AREA)
                        self.frame_queue.put(frame, timestamp=time.time())
            except Exception:
                pass
            elapsed = time.perf_counter() - t0
            sleep_t = max(0.01, 0.033 - elapsed)
            time.sleep(sleep_t)

    def get_latest_frame(self, timeout: float = 0.5) -> Tuple[Optional[np.ndarray], float]:
        """Retrieves newest frame with zero lag."""
        return self.frame_queue.get_latest(timeout=timeout)

    def stop(self) -> None:
        """Shuts down stream connection."""
        with self._lock:
            self._running = False
            self._connected = False
            if self._socket:
                try:
                    self._socket.close()
                except Exception:
                    pass
                self._socket = None

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
