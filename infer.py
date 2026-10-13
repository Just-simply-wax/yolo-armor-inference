#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO26n-pose 装甲板视频推理与关键点可视化

流水线（对应作业第三节的四条功能）：
    1) 加载模型权重      YOLO(weights)
    2) 读取视频          cv2.VideoCapture
    3) 逐帧推理          model.predict(frame, ...)
    4) 绘制结果          检测框 / 类别 / 置信度 / 模型预测的 4 个关键点
    5) 写出视频          OpenCV 写 mp4v -> ffmpeg 转 H.264（保证通用播放器能打开）

用法：
    python infer.py --weights weights/best.pt --source data/armor_raw.mp4 \
        --output runs/result.mp4 --conf 0.25 --imgsz 640 --device 0 --save-frames assets
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# 按类别编号分配颜色，保证不同类别的框/关键点颜色不同，便于区分归属
PALETTE = [
    (0, 255, 0), (0, 128, 255), (255, 0, 0),
    (255, 0, 255), (0, 255, 255), (255, 255, 0),
]


def parse_args() -> argparse.Namespace:
    """命令行参数：模型/视频/输出路径与推理参数都可从外部修改。"""
    p = argparse.ArgumentParser(
        description="装甲板视频推理：检测框 + 类别 + 置信度 + 4 个关键点"
    )
    p.add_argument("--weights", default="weights/best.pt", help="模型权重路径")
    p.add_argument("--source", default="data/armor_raw.mp4", help="输入视频路径")
    p.add_argument("--output", default="runs/result.mp4", help="输出视频路径（H.264）")
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸")
    p.add_argument("--device", default="0", help="0=第一块 GPU；cpu=CPU")
    p.add_argument("--save-frames", default="", help="代表性帧保存目录（可留空）")
    p.add_argument("--max-frames", type=int, default=0, help="只处理前 N 帧（0=全部，调试用）")
    return p.parse_args()


def fourcc(code: str) -> int:
    """兼容不同 OpenCV 版本获取 fourcc 编码标识。"""
    if hasattr(cv2, "VideoWriter_fourcc"):
        return cv2.VideoWriter_fourcc(*code)
    return cv2.VideoWriter.fourcc(*code)


def pick_ffmpeg():
    """优先用系统 ffmpeg；没有则退回 imageio-ffmpeg 自带的可执行文件。"""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def to_h264(tmp_path: Path, out_path: Path) -> bool:
    """把 OpenCV 写出的 mp4v 转成通用可播的 H.264 / yuv420p 视频。"""
    ff = pick_ffmpeg()
    if ff is None:
        shutil.move(str(tmp_path), str(out_path))
        print("[warn] 未找到 ffmpeg，保留 mp4v 编码（部分播放器可能打不开）")
        return False
    cmd = [
        ff, "-y", "-i", str(tmp_path),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",   # libx264 + yuv420p 要求宽高为偶数
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tmp_path.unlink(missing_ok=True)
    return True


def draw_detections(frame, result, names) -> tuple:
    """绘制检测框、类别、置信度与模型预测的关键点，返回 (目标数, 含4关键点的目标数)。"""
    n_obj, n_kpt = 0, 0
    kpts = result.keypoints                      # 模型 pose 头的输出
    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        cid = int(box.cls.item())
        conf = float(box.conf.item())
        color = PALETTE[cid % len(PALETTE)]

        # 1) 检测框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 2) 类别 + 置信度（先铺底色，亮背景下也看得清）
        label = f"{names[cid]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        # 3) 关键点：来自模型预测 result.keypoints.xy（形状 [N,4,2]），与 boxes 逐实例对应
        #    注意：绝不能用检测框的四个角代替，那属于伪造关键点
        if kpts is not None and len(kpts.xy) > i:
            xy = kpts.xy[i].cpu().numpy()        # (4, 2) 像素坐标
            if len(xy) >= 4:
                n_kpt += 1
                cv2.polylines(frame, [xy.astype(np.int32)], True, color, 2, cv2.LINE_AA)
            for j, (px, py) in enumerate(xy):
                cv2.circle(frame, (int(px), int(py)), 4, color, -1, cv2.LINE_AA)
                cv2.putText(frame, f"P{j+1}", (int(px) + 6, int(py) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        n_obj += 1
    return n_obj, n_kpt


def main() -> None:
    args = parse_args()
    w_path, s_path = Path(args.weights), Path(args.source)
    if not w_path.exists():
        sys.exit(f"[error] 找不到权重文件：{w_path}")
    if not s_path.exists():
        sys.exit(f"[error] 找不到输入视频：{s_path}")

    # ---------- 1) 加载模型 ----------
    model = YOLO(str(w_path))
    kpt_shape = getattr(getattr(model, "model", None), "kpt_shape", None)
    print(f"[info] 模型 task={model.task} kpt_shape={kpt_shape}")
    print(f"[info] 类别 {model.names}")

    # ---------- 2) 打开视频 ----------
    cap = cv2.VideoCapture(str(s_path))
    if not cap.isOpened():
        sys.exit(f"[error] 无法打开视频：{s_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 1:        # 0 / NaN / 异常值兜底
        print(f"[warn] 读到的 fps={fps} 异常，改用 30")
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[info] 视频 {w}x{h} @ {fps:.2f}fps，约 {total} 帧")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.save_frames:
        Path(args.save_frames).mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f"_tmp_{out_path.stem}.mp4")

    writer = cv2.VideoWriter(str(tmp_path), fourcc("mp4v"), fps, (w, h))
    if not writer.isOpened():
        sys.exit("[error] VideoWriter 打开失败（检查输出目录是否可写）")

    counts = Counter()
    conf_sum = 0.0
    frames_with_det = 0
    instances_with_kpt = 0
    idx = 0
    saved = 0
    t0 = time.time()

    # ---------- 3) 逐帧推理 + 4) 绘制 ----------
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                               device=args.device, verbose=False)[0]
        n_obj, n_kpt = draw_detections(frame, result, model.names)

        if n_obj:
            frames_with_det += 1
            instances_with_kpt += n_kpt
            conf_sum += float(result.boxes.conf.sum().item())
            for c in result.boxes.cls.tolist():
                counts[model.names[int(c)]] += 1
            if args.save_frames and saved < 3 and idx % 10 == 0:
                cv2.imwrite(str(Path(args.save_frames) / f"frame_{idx:05d}.jpg"), frame)
                saved += 1

        # 左上角状态栏：便于确认这段视频确实是模型推理输出
        hud = f"{w_path.name} | frame {idx}/{total or '?'} | objects {n_obj} | conf>={args.conf}"
        cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(frame)
        if idx % 30 == 0:
            print(f"  进度 {idx}/{total or '?'} 帧，累计检出 {sum(counts.values())} 个目标")
        if args.max_frames and idx >= args.max_frames:
            break

    cap.release()
    writer.release()

    # ---------- 5) 转码输出 ----------
    used_h264 = to_h264(tmp_path, out_path)

    elapsed = time.time() - t0
    stats = {
        "weights": str(w_path),
        "source": str(s_path),
        "output": str(out_path),
        "video_size": f"{w}x{h}",
        "video_fps": round(fps, 2),
        "frames_processed": idx,
        "frames_with_detection": frames_with_det,
        "total_objects": sum(counts.values()),
        "instances_with_4_keypoints": instances_with_kpt,
        "per_class": dict(counts),
        "avg_conf": round(conf_sum / max(1, sum(counts.values())), 4),
        "elapsed_sec": round(elapsed, 2),
        "avg_fps": round(idx / elapsed, 2) if elapsed else 0,
        "device": args.device,
        "conf_thres": args.conf,
        "imgsz": args.imgsz,
        "output_h264": used_h264,
    }
    print("[result] " + json.dumps(stats, ensure_ascii=False, indent=2))
    stats_path = out_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] 结果视频：{out_path}")
    print(f"[done] 统计文件：{stats_path}")


if __name__ == "__main__":
    main()
