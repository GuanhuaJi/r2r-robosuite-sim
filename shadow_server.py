#!/usr/bin/env python3
"""
Real-time "shadow" server.

- Start with a dataset (one of: can, lift, square, stack, three_piece_assembly)
  and a list of robots (e.g., Panda UR5e Jaco Sawyer Kinova3 IIWA).
- Listen on a TCP port.
- For each request: receive image (84x84x3 uint8), eef_pose (7 float32), gripper (1 float32)
- Compute union of robot masks (each robot computed in its own process to avoid EGL pollution)
- Overlay union as black onto the input image
- Send back the resulting image (84x84x3 uint8)

Wire protocol (request):
  - 4 bytes big-endian uint32: header_len
  - header JSON bytes (utf-8): {"H":84,"W":84,"C":3}
  - image bytes: H*W*C uint8 (row-major)
  - pose bytes: 7 float32 (little-endian)
  - grip bytes: 1 float32 (little-endian)

Wire protocol (response):
  - 4 bytes big-endian uint32: header_len
  - header JSON bytes: {"H":84,"W":84,"C":3}
  - image bytes: H*W*C uint8
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import multiprocessing as mp


# ---------------------- project imports (worker will import envs lazily) ----------------------
from core import pick_best_gpu
from config.robot_pose_dict import ROBOT_POSE_DICT


# ---------------------- constants ----------------------
DEFAULT_H = 84
DEFAULT_W = 84
DEFAULT_C = 3


def select_gripper(robot: str) -> str:
    if robot == "Sawyer":
        return "RethinkGripper"
    if robot == "Jaco":
        return "JacoThreeFingerGripper"
    if robot in {"IIWA", "UR5e", "Kinova3"}:
        return "Robotiq85Gripper"
    if robot == "Panda":
        return "PandaGripper"
    raise ValueError(f"Unknown robot {robot!r}")


def _read_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise EOFError("connection closed")
        buf.extend(chunk)
    return bytes(buf)


def _send_all(conn: socket.socket, data: bytes) -> None:
    view = memoryview(data)
    while view:
        sent = conn.send(view)
        view = view[sent:]


def _recv_request(conn: socket.socket) -> Tuple[np.ndarray, np.ndarray, float]:
    # header
    header_len = struct.unpack(">I", _read_exact(conn, 4))[0]
    header = json.loads(_read_exact(conn, header_len).decode("utf-8"))
    H = int(header["H"])
    W = int(header["W"])
    C = int(header.get("C", 3))

    # image
    img_n = H * W * C
    img_bytes = _read_exact(conn, img_n)
    img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(H, W, C)

    # pose (7 float32)
    pose_bytes = _read_exact(conn, 7 * 4)
    pose = np.frombuffer(pose_bytes, dtype=np.float32).copy()  # copy: avoid referencing socket buffer

    # grip (1 float32)
    grip_bytes = _read_exact(conn, 4)
    grip = float(np.frombuffer(grip_bytes, dtype=np.float32)[0])

    return img, pose, grip


def _send_response(conn: socket.socket, img: np.ndarray) -> None:
    img = np.asarray(img, dtype=np.uint8)
    H, W, C = img.shape
    header = json.dumps({"H": H, "W": W, "C": C}).encode("utf-8")
    _send_all(conn, struct.pack(">I", len(header)))
    _send_all(conn, header)
    _send_all(conn, img.tobytes(order="C"))


# ---------------------- worker process ----------------------
@dataclass
class WorkerCfg:
    robot: str
    robot_dataset: str
    H: int
    W: int


def _pack_mask_bool(mask_bool: np.ndarray) -> bytes:
    # packbits on flattened array -> bytes
    flat = mask_bool.reshape(-1).astype(np.uint8)
    packed = np.packbits(flat, bitorder="little")
    return packed.tobytes()


def _unpack_mask_bool(packed_bytes: bytes, H: int, W: int) -> np.ndarray:
    packed = np.frombuffer(packed_bytes, dtype=np.uint8)
    flat = np.unpackbits(packed, bitorder="little")[: H * W].astype(bool)
    return flat.reshape(H, W)


def worker_loop(cfg: WorkerCfg, conn: mp.connection.Connection) -> None:
    """
    Each worker owns exactly ONE TargetEnvWrapper + EGL context.
    It receives (pose_bytes, grip_float32) and returns packed mask bytes.
    """
    try:
        pick_best_gpu()
        os.environ["MUJOCO_GL"] = "egl"

        # import here (per-process)
        from envs import TargetEnvWrapper

        gripper_name = select_gripper(cfg.robot)
        robot_disp = np.asarray(ROBOT_POSE_DICT[cfg.robot_dataset][cfg.robot], dtype=np.float32)

        wrapper = TargetEnvWrapper(
            cfg.robot,
            gripper_name,
            cfg.robot_dataset,
            camera_height=cfg.H,
            camera_width=cfg.W,
        )

        while True:
            msg = conn.recv()
            if msg is None:
                break

            pose_bytes, grip_f32 = msg
            pose = np.frombuffer(pose_bytes, dtype=np.float32).reshape(7)
            grip_val = float(np.float32(grip_f32))

            ok, _rgb, mask = wrapper.generate_one_image(
                eef_pose=pose,
                gripper_value=grip_val,
                robot_dataset=cfg.robot_dataset,
                robot_disp=robot_disp,
            )
            if not ok:
                conn.send(("FAIL", b""))
                continue

            # mask is uint8 {0,255}
            mask = np.asarray(mask)
            if mask.ndim == 3:
                mask = mask[..., 0]
            mask_bool = (mask > 127)

            conn.send(("OK", _pack_mask_bool(mask_bool)))

        wrapper.target_env.env.close_renderer()

    except Exception as e:
        try:
            conn.send(("ERR", str(e).encode("utf-8", errors="ignore")))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ---------------------- server ----------------------
def run_server(
    host: str,
    port: int,
    robot_dataset: str,
    robots: List[str],
    H: int,
    W: int,
) -> None:
    # spawn workers
    ctx = mp.get_context("spawn")

    worker_conns_parent: Dict[str, mp.connection.Connection] = {}
    procs: List[mp.Process] = []

    for r in robots:
        parent_conn, child_conn = ctx.Pipe()
        cfg = WorkerCfg(robot=r, robot_dataset=robot_dataset, H=H, W=W)
        p = ctx.Process(target=worker_loop, args=(cfg, child_conn), daemon=True)
        p.start()
        child_conn.close()  # parent doesn't use child end
        worker_conns_parent[r] = parent_conn
        procs.append(p)

    # listen socket
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(8)

    print(f"[shadow_server] dataset={robot_dataset} robots={robots} listen={host}:{port} HW={H}x{W}")

    try:
        while True:
            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[shadow_server] client connected: {addr}")

            try:
                while True:
                    img, pose, grip = _recv_request(conn)

                    # union mask in packed-bit form (fast)
                    pose_bytes = np.asarray(pose, dtype=np.float32).tobytes(order="C")
                    grip_f32 = np.float32(grip)

                    packed_union = None

                    for r in robots:
                        worker_conns_parent[r].send((pose_bytes, grip_f32))

                    for r in robots:
                        status, payload = worker_conns_parent[r].recv()
                        if status == "OK":
                            packed = np.frombuffer(payload, dtype=np.uint8)
                            if packed_union is None:
                                packed_union = packed.copy()
                            else:
                                np.bitwise_or(packed_union, packed, out=packed_union)
                        elif status == "FAIL":
                            # treat as empty mask
                            if packed_union is None:
                                # need correct length: ceil(H*W/8)
                                nbytes = (H * W + 7) // 8
                                packed_union = np.zeros((nbytes,), dtype=np.uint8)
                        else:
                            # ERR: worker crashed; propagate a simple empty mask
                            if packed_union is None:
                                nbytes = (H * W + 7) // 8
                                packed_union = np.zeros((nbytes,), dtype=np.uint8)

                    if packed_union is None:
                        nbytes = (H * W + 7) // 8
                        packed_union = np.zeros((nbytes,), dtype=np.uint8)

                    union_mask = _unpack_mask_bool(packed_union.tobytes(), H, W)

                    out = img.copy()
                    out[union_mask] = 0  # overlay as black

                    _send_response(conn, out)

            except EOFError:
                print(f"[shadow_server] client disconnected: {addr}")
            except Exception as e:
                print(f"[shadow_server] client error: {e}", file=sys.stderr)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

    except KeyboardInterrupt:
        print("\n[shadow_server] shutting down...")

    finally:
        try:
            srv.close()
        except Exception:
            pass

        # stop workers
        for r, pc in worker_conns_parent.items():
            try:
                pc.send(None)
            except Exception:
                pass
            try:
                pc.close()
            except Exception:
                pass

        for p in procs:
            try:
                p.join(timeout=2.0)
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["can", "lift", "square", "stack", "three_piece_assembly"])
    ap.add_argument("--robots", nargs="+", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--H", type=int, default=DEFAULT_H)
    ap.add_argument("--W", type=int, default=DEFAULT_W)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_server(
        host=args.host,
        port=args.port,
        robot_dataset=args.dataset,
        robots=args.robots,
        H=int(args.H),
        W=int(args.W),
    )


if __name__ == "__main__":
    main()

'''
python /home/guanhuaji/oxe_auge_sim_augmentation/r2r-robosuite/shadow_server.py --dataset can --robots Panda UR5e --port 5555

python shadow_server.py --dataset can --robots Panda UR5e --port 5555

'''