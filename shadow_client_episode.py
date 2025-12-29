#!/usr/bin/env python3
"""
Episode client for shadow_server.py

- Reads an entire episode from:
    VIDEO_PATH  (frames, 84x84x3)
    NPZ_PATH    (pos(T,7), grip(T,))
- For each frame i:
    send (frame_i, pos[i], grip[i]) to the shadow server
    receive shadow frame (84x84x3)
- Writes all received frames into one MP4 via imageio

Protocol matches the server:
Request:
  u32_be header_len
  header json bytes: {"H":84,"W":84,"C":3}
  image bytes: H*W*C uint8
  pose bytes: 7 float32 (little-endian)
  grip bytes: 1 float32 (little-endian)

Response:
  u32_be header_len
  header json bytes: {"H":84,"W":84,"C":3}
  image bytes: H*W*C uint8
"""

import json
import socket
import struct
import os
from pathlib import Path

import numpy as np
import imageio.v2 as iio
from tqdm import tqdm


# ---------------------- user-editable variables ----------------------
HOST = "127.0.0.1"
PORT = 5555

NPZ_PATH = "/home/guanhuaji/oxe_auge_sim_augmentation/r2r-robosuite/example/0.npz"
VIDEO_PATH = "/home/guanhuaji/oxe_auge_sim_augmentation/r2r-robosuite/example/0.mp4"

OUT_MP4 = "/home/guanhuaji/oxe_auge_sim_augmentation/r2r-robosuite/example/shadow_out_0.mp4"

H = 84
W = 84
C = 3
# --------------------------------------------------------------------


def read_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError("connection closed")
        buf.extend(chunk)
    return bytes(buf)


def send_all(sock: socket.socket, data: bytes) -> None:
    view = memoryview(data)
    while view:
        sent = sock.send(view)
        view = view[sent:]


def main():
    # load trajectory
    data = np.load(NPZ_PATH, allow_pickle=True)
    pos = data["pos"]
    grip = data["grip"]

    if pos.ndim != 2 or pos.shape[1] != 7:
        raise ValueError(f"pos must be (T,7), got {pos.shape}")
    T_npz = int(pos.shape[0])
    if grip.shape[0] != T_npz:
        raise ValueError(f"grip length {grip.shape[0]} != pos length {T_npz}")

    # open video
    reader = iio.get_reader(VIDEO_PATH)
    meta = reader.get_meta_data()
    fps = meta.get("fps", 30)

    T_video = None
    try:
        T_video = int(reader.count_frames())
    except Exception:
        T_video = None

    T_total = T_npz if T_video is None else min(T_npz, T_video)

    # constant header (we still send it per-frame, because the server expects it each request)
    header = json.dumps({"H": H, "W": W, "C": C}).encode("utf-8")
    header_prefix = struct.pack(">I", len(header)) + header

    out_path = Path(OUT_MP4)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # connect once, stream requests/responses
    with socket.create_connection((HOST, PORT)) as s:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        with iio.get_writer(
            str(out_path),
            fps=fps,
            codec="libx264",
            macro_block_size=1,
            pixelformat="yuv420p",
        ) as writer:
            for i in tqdm(range(T_total), desc="Shadow client (episode)", dynamic_ncols=True):
                frame = np.asarray(reader.get_data(i)).astype(np.uint8)

                # normalize to (H,W,3)
                if frame.ndim == 2:
                    frame = np.repeat(frame[..., None], 3, axis=2)
                if frame.shape[2] > 3:
                    frame = frame[..., :3]

                if frame.shape != (H, W, C):
                    raise ValueError(f"Video frame shape {frame.shape} != {(H, W, C)} at i={i}")

                pose_i = np.asarray(pos[i], dtype=np.float32).reshape(7)
                grip_i = np.float32(np.asarray(grip[i]).reshape(-1)[0])

                # ---- send request ----
                send_all(s, header_prefix)
                send_all(s, frame.tobytes(order="C"))
                send_all(s, pose_i.tobytes(order="C"))
                send_all(s, grip_i.tobytes(order="C"))

                # ---- recv response ----
                hdr_len = struct.unpack(">I", read_exact(s, 4))[0]
                _hdr = json.loads(read_exact(s, hdr_len).decode("utf-8"))
                H2, W2, C2 = int(_hdr["H"]), int(_hdr["W"]), int(_hdr["C"])
                out_bytes = read_exact(s, H2 * W2 * C2)
                out_frame = np.frombuffer(out_bytes, dtype=np.uint8).reshape(H2, W2, C2)

                writer.append_data(out_frame)

    reader.close()
    print(f"[OK] wrote shadow mp4: {out_path}")


if __name__ == "__main__":
    main()

'''
python /home/guanhuaji/oxe_auge_sim_augmentation/r2r-robosuite/shadow_client_episode.py
'''