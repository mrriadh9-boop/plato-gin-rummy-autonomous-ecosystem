"""
Synthetic H.264 Stream Generator and Mock Scrcpy TCP Server.
Used for offline testing, CI verification, and hardware-in-the-loop simulation.
"""
from __future__ import annotations

import io
import socket
import struct
import threading
import time
from typing import Callable, Generator, List, Optional, Tuple, Union

import av
import cv2
import numpy as np


def encode_frames_to_h264(
    frames: List[np.ndarray],
    fps: int = 30,
    bitrate: int = 6000000,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> bytes:
    """
    Encodes a list of BGR24 numpy frames into raw H.264 Annex-B byte stream using PyAV.
    """
    if not frames:
        return b""

    h, w = frames[0].shape[:2]
    out_w = width or w
    out_h = height or h

    # H.264 dimensions must be divisible by 2
    out_w = (out_w // 2) * 2
    out_h = (out_h // 2) * 2

    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="h264")
    stream = container.add_stream("h264", rate=fps)
    stream.width = out_w
    stream.height = out_h
    stream.pix_fmt = "yuv420p"
    stream.bit_rate = bitrate

    for frame in frames:
        if frame.shape[0] != out_h or frame.shape[1] != out_w:
            frame_resized = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        else:
            frame_resized = frame

        vf = av.VideoFrame.from_ndarray(frame_resized, format="bgr24")
        for packet in stream.encode(vf):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()
    return buf.getvalue()


def encode_single_frame_to_h264(frame: np.ndarray, bitrate: int = 6000000) -> bytes:
    """Encodes a single BGR24 numpy frame into H.264 Annex-B bytes."""
    return encode_frames_to_h264([frame], fps=30, bitrate=bitrate)


def create_scrcpy_header(
    device_name: str = "Plato-Tablet-1800x2880",
    width: int = 1800,
    height: int = 2880,
    codec_id: bytes = b"h264",
) -> bytes:
    """
    Constructs the canonical Scrcpy handshake byte header:
    - 1 dummy byte (0x00)
    - 64 bytes device name (null-padded utf-8)
    - 4 bytes codec ID (b'h264' / 0x68323634)
    - 4 bytes width (big endian uint32)
    - 4 bytes height (big endian uint32)
    """
    dummy = b"\x00"
    dev_bytes = device_name.encode("utf-8")[:64].ljust(64, b"\x00")
    codec = codec_id[:4].ljust(4, b"\x00")
    dims = struct.pack(">II", width, height)
    return dummy + dev_bytes + codec + dims


class MockScrcpyServer:
    """
    Lightweight local TCP socket server emulating a Scrcpy device daemon.
    Streams synthetic H.264 frames to connecting Scrcpy clients.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        width: int = 1800,
        height: int = 2880,
        fps: int = 30,
    ):
        self.host = host
        self.port = port
        self.width = width
        self.height = height
        self.fps = fps
        self.server_sock: Optional[socket.socket] = None
        self.actual_port: int = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._client_sockets: List[socket.socket] = []
        self._lock = threading.Lock()
        self._frame_supplier: Optional[Callable[[], np.ndarray]] = None

    def set_frame_supplier(self, supplier: Callable[[], np.ndarray]) -> None:
        """Sets a callable that supplies dynamic BGR frames on demand."""
        self._frame_supplier = supplier

    def start(self) -> int:
        """Starts the server socket in a background daemon thread and returns bound port."""
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(5)
        self.actual_port = self.server_sock.getsockname()[1]
        self._running = True

        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()
        return self.actual_port

    def _accept_loop(self) -> None:
        while self._running:
            try:
                if self.server_sock is None:
                    break
                self.server_sock.settimeout(0.5)
                client_sock, _ = self.server_sock.accept()
                with self._lock:
                    self._client_sockets.append(client_sock)
                # Handle client in a thread
                t = threading.Thread(target=self._stream_to_client, args=(client_sock,), daemon=True)
                t.start()
            except socket.timeout:
                continue
            except Exception:
                break

    def _stream_to_client(self, client_sock: socket.socket) -> None:
        try:
            # 1. Send Scrcpy header
            header = create_scrcpy_header(width=self.width, height=self.height)
            client_sock.sendall(header)

            interval = 1.0 / self.fps
            frame_idx = 0

            while self._running:
                t0 = time.perf_counter()

                # Get frame
                if self._frame_supplier:
                    frame = self._frame_supplier()
                else:
                    # Generate default synthetic test pattern
                    frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    cv2.putText(
                        frame,
                        f"Plato Scrcpy Mock Frame {frame_idx}",
                        (100, 300),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.0,
                        (0, 255, 0),
                        3,
                    )

                h264_chunk = encode_single_frame_to_h264(frame)
                if h264_chunk:
                    # Send packet header + payload
                    pts = int(time.time() * 1000)
                    pkt_header = struct.pack(">QI", pts, len(h264_chunk))
                    client_sock.sendall(pkt_header + h264_chunk)

                frame_idx += 1
                elapsed = time.perf_counter() - t0
                sleep_time = max(0.0, interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except (socket.error, ConnectionResetError, BrokenPipeError):
            pass
        finally:
            with self._lock:
                if client_sock in self._client_sockets:
                    self._client_sockets.remove(client_sock)
            try:
                client_sock.close()
            except Exception:
                pass

    def stop(self) -> None:
        """Stops the mock server and disconnects all clients."""
        self._running = False
        with self._lock:
            for s in self._client_sockets:
                try:
                    s.close()
                except Exception:
                    pass
            self._client_sockets.clear()

        if self.server_sock:
            try:
                self.server_sock.close()
            except Exception:
                pass
            self.server_sock = None

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)


class SyntheticStreamGenerator:
    """
    In-process synthetic video stream generator providing get_latest_frame().
    Loads ground-truth Plato frames or renders synthetic hands for offline testing.
    """

    def __init__(
        self,
        fps: float = 30.0,
        width: int = 1800,
        height: int = 2880,
        asset_dir: Optional[str] = None,
    ):
        self.fps = fps
        self.width = width
        self.height = height
        self._running = False
        self._frame_idx = 0
        self._frames: List[np.ndarray] = []
        self._load_assets(asset_dir)

    def _load_assets(self, asset_dir: Optional[str] = None):
        import os, glob
        if asset_dir is None:
            proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            asset_dir = os.path.join(proj_root, "test_assets", "live_tablet_captures")
            if not os.path.exists(asset_dir):
                asset_dir = os.path.join(proj_root, "test_assets")

        if os.path.exists(asset_dir):
            for ext in ("*.png", "*.jpg"):
                for p in glob.glob(os.path.join(asset_dir, "**", ext), recursive=True):
                    img = cv2.imread(p)
                    if img is not None:
                        self._frames.append(img)

    def start(self) -> bool:
        self._running = True
        return True

    def stop(self) -> None:
        self._running = False

    @property
    def is_connected(self) -> bool:
        return self._running

    def get_latest_frame(self, timeout: float = 0.5) -> Tuple[Optional[np.ndarray], float]:
        """Returns the next sample frame or renders a synthetic game table."""
        now = time.time()
        if self._frames:
            idx = self._frame_idx % len(self._frames)
            self._frame_idx += 1
            return self._frames[idx].copy(), now

        # Generate procedural synthetic table frame
        frame = np.full((self.height, self.width, 3), (28, 38, 24), dtype=np.uint8)
        cv2.putText(
            frame,
            f"PLATO SYNTHETIC BENCHMARK - Frame {self._frame_idx}",
            (100, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            (0, 220, 100),
            3,
        )
        self._frame_idx += 1
        return frame, now
