
"""
Windows 离线说话人分离 - pyannote.audio 4.0.4 + community-1
使用 soundfile 读取音频，绕过 torchcodec
"""
import os
import sys
FFMPEG_BIN = r"E:\ffmpeg\bin"
if os.path.isdir(FFMPEG_BIN):
    os.add_dll_directory(FFMPEG_BIN)

import logging
import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Pipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5.5s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def separate_speakers(audio_path: str, output_dir: str, num_speakers: int = None) -> bool:
    log.info("===== 开始说话人分离 =====")
    log.info(f"输入文件: {audio_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"使用设备: {device}")

    # ---------- 加载 Pipeline ----------
    try:
        log.info("正在加载 community-1 ...")
        pipeline = Pipeline.from_pretrained(
            r"models/speaker-diarization-community-1"

        )
        pipeline.to(device)
        log.info("✅ Pipeline 加载成功")
    except Exception as e:
        log.error(f"❌ Pipeline 加载失败: {e}", exc_info=True)
        return False

    # ---------- 用 soundfile 加载音频，避免 torchcodec ----------
    try:
        data, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        # data 形状为 (time, channels)，取第一声道并转成 (1, time)
        waveform = torch.from_numpy(data[:, 0]).unsqueeze(0)
        duration = waveform.shape[1] / sample_rate
        log.info(f"音频已加载: {sample_rate}Hz, {duration:.2f}s, 形状={tuple(waveform.shape)}")
    except Exception as e:
        log.error(f"❌ 音频加载失败: {e}", exc_info=True)
        return False

    # ---------- 执行说话人日志（传入内存数据） ----------
    try:
        pipeline_kwargs = {}
        if num_speakers is not None:
            pipeline_kwargs["num_speakers"] = num_speakers

        output = pipeline(
            audio_path
        )
        diarization = output.speaker_diarization
        log.info("✅ 说话人日志分析完成")
    except Exception as e:
        log.error(f"❌ 分析失败: {e}", exc_info=True)
        if "CUDA out of memory" in str(e):
            log.error("提示: CUDA 显存不足，请尝试更短音频或使用 CPU")
        return False

    # ---------- 创建输出目录 ----------
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        log.error(f"❌ 创建输出目录失败: {e}", exc_info=True)
        return False

    # ---------- 按说话人分离音轨 ----------
    speakers = diarization.labels()
    num_detected = len(speakers)
    log.info(f"检测到 {num_detected} 位说话人: {list(speakers)}")

    if num_detected == 0:
        log.warning("⚠️ 未检测到任何说话人活动")
        return True

    if num_speakers is not None and num_detected != num_speakers:
        log.warning(f"⚠️ 检测数({num_detected})与指定数({num_speakers})不一致，按检测结果继续")

    waveform = waveform.to(device)
    separated = {spk: torch.zeros_like(waveform) for spk in speakers}

    total_segments = sum(1 for _ in diarization.itertracks(yield_label=True))
    processed = 0

    log.info("开始分离音轨...")
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_sample = int(turn.start * sample_rate)
        end_sample = min(int(turn.end * sample_rate), waveform.shape[1])
        start_sample = min(start_sample, end_sample)

        if speaker in separated:
            separated[speaker][:, start_sample:end_sample] = waveform[:, start_sample:end_sample]

        processed += 1
        if processed % 100 == 0 or processed == total_segments:
            log.info(f"  进度: {processed}/{total_segments}")

    log.info("✅ 音轨分离完成")

    # ---------- 保存文件（用 soundfile） ----------
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    save_ok = True

    for speaker, wave in separated.items():
        out_path = os.path.join(output_dir, f"{base_name}_speaker_{speaker}.wav")
        try:
            sf.write(out_path, wave.squeeze(0).cpu().numpy(), sample_rate)
            log.info(f"  已保存: {out_path}")
        except Exception as e:
            log.error(f"❌ 保存失败 {out_path}: {e}", exc_info=True)
            save_ok = False

    if save_ok:
        log.info("===== ✅ 处理成功完成 =====")
    else:
        log.error("===== ⚠️ 处理完成，但部分文件保存失败 =====")

    return save_ok


if __name__ == "__main__":
    INPUT_AUDIO = r"会议录音.wav"
    OUTPUT_DIR = r".\output"
    NUM_SPEAKERS = None

    if not os.path.exists(INPUT_AUDIO):
        log.error(f"❌ 输入文件不存在: {INPUT_AUDIO}")
        sys.exit(1)

    success = separate_speakers(
        audio_path=INPUT_AUDIO,
        output_dir=OUTPUT_DIR,
        num_speakers=NUM_SPEAKERS,
    )

    sys.exit(0 if success else 1)
