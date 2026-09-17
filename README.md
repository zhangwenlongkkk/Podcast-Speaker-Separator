# 播客语音分离器 (Podcast Speaker Separator)

本项目包含一个 Python 脚本，用于分离双人（或多人）对话播客音频文件中的不同说话人语音。它利用 `pyannote.audio` 库进行说话人日志分析（Speaker Diarization），找出“谁在什么时候说话”，并将每个说话人的语音片段提取到单独的音轨中。

关键特性是：**输出的每个音轨都与原始音频文件具有相同的时间长度，在对应说话人未讲话的时间段填充静音**。这对于需要保持时间轴对齐的应用（如字幕制作、后续分析）非常有用。

## 功能

*   自动检测音频中的说话人及其讲话时间段。
*   为每个检测到的说话人生成一个独立的音频文件。
*   输出的音频文件与原始文件等长，非语音部分填充静音，保持时间轴同步。
*   支持 GPU 加速（如果 CUDA 环境可用）。
*   可以通过命令行参数配置输入/输出路径和 Hugging Face Token。
*   （可选）可以指定预期的说话人数量以辅助模型。

## 环境要求

1.  **Python**: 推荐 3.10 或更高版本（`pyannote.audio` 4.x 要求 Python ≥ 3.10）。
2.  **pip**: Python 包管理器。
3.  **ffmpeg**: 用于处理多媒体文件的工具。`pyannote.audio` 4.x 通过 `torchcodec` 调用 FFmpeg 解码音频。
    *   **Linux (Ubuntu/Debian)**: `sudo apt update && sudo apt install ffmpeg`
    *   **macOS (Homebrew)**: `brew install ffmpeg`
    *   **Windows**: 见下方 [Windows 特别注意事项](#windows-特别注意事项)。**务必下载 shared 版本**，不要用 static 版本。
4.  **Hugging Face 账户**: 需要访问 [Hugging Face](https://huggingface.co/) 注册账户。
5.  **Hugging Face 访问令牌 (Token)**:
    *   在 Hugging Face 网站的用户设置 → Access Tokens 中创建一个具有 `read` 权限的令牌。
    *   可以通过 `huggingface-cli login` 登录，或在运行脚本时通过 `--token` 参数提供。
6.  **同意模型使用条款**: `pyannote.audio` 的预训练模型托管在 Hugging Face Hub 上。需要访问以下模型页面并同意其使用条款：
    *   **主要模型**: [pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1)
    *   **依赖模型**: [pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0)
    *   **依赖模型**: [pyannote/embedding](https://huggingface.co/pyannote/embedding)
    *   *具体依赖可能随模型版本变化。如遇加载错误，请查看模型卡片和错误信息。*

## Windows 特别注意事项

Windows 下 `pyannote.audio` 4.x 最常见的坑都集中在 `torchcodec` 与 FFmpeg 上。你当前脚本里已经处理了其中一步，但还有几个细节需要确认。

### 1. FFmpeg 必须使用 shared 版本

`torchcodec` 需要的是 FFmpeg 的**动态链接库（DLL）**，而不是静态链接的 `ffmpeg.exe`。

*   **错误示范**：下载 `ffmpeg-n7.x-win64-gpl-7.x.zip`（静态版）。命令行能用，但 `torchcodec` 找不到 DLL。
*   **正确做法**：下载文件名包含 **`shared`** 的版本，例如：
    *   `ffmpeg-n7.1.x-win64-gpl-shared-7.1.zip`
    *   下载地址：<https://www.gyan.dev/ffmpeg/builds/> 或 <https://github.com/BtbN/FFmpeg-Builds/releases>
*   解压后，记住 `bin` 目录的完整路径，例如 `E:\ffmpeg\bin`。

### 2. 在脚本最顶部注册 DLL 目录

你当前脚本里已经这样做了，位置是正确的：

```python
import os
import sys

FFMPEG_BIN = r"E:\ffmpeg\bin"
if os.path.isdir(FFMPEG_BIN):
    os.add_dll_directory(FFMPEG_BIN)

# 然后再导入其他库
import torch
from pyannote.audio import Pipeline
```

**关键点**：`os.add_dll_directory` 必须在任何可能触发 `torchcodec` 加载的库（如 `pyannote.audio`）之前调用。你脚本里的顺序是对的。

### 3. 验证 torchcodec 是否加载成功

在命令行单独跑一句：

```python
import torchcodec
print(torchcodec.__version__)
```

如果输出版本号，说明 DLL 已经通了。如果报 `Could not load libtorchcodec`，说明 FFmpeg DLL 仍未正确加载，检查：

*   `E:\ffmpeg\bin` 下是否确实有 `avcodec-*.dll`、`avformat-*.dll`、`avutil-*.dll` 等文件。
*   这些 DLL 是不是 shared 版本解压出来的。
*   `FFMPEG_BIN` 路径有没有写错。

### 4. 备选方案：手动复制 DLL

如果添加目录后仍报错，可以把 FFmpeg `bin` 目录下**所有 `.dll` 文件**复制到 `torchcodec` 的安装目录：

```
C:\Users\<用户名>\.conda\envs\<环境名>\Lib\site-packages\torchcodec\
```

这是很多 Windows 用户的最终解决方案，能强制 `torchcodec` 在同一目录下找到依赖。

## 脚本当前行为说明

你当前脚本里有几个地方需要特别注意，它们直接影响能否成功运行。

### 1. 模型加载方式：本地目录直接加载

脚本里写的是：

```python
pipeline = Pipeline.from_pretrained(
    r"models/speaker-diarization-community-1"
)
```

这是**本地目录直接加载**模式，要求 `models/speaker-diarization-community-1/` 下面直接是 `config.yaml`、`pytorch_model.bin` 这类文件。

如果磁盘上实际是 Hugging Face cache 结构：

```
models/
  models--pyannote--speaker-diarization-community-1/
    snapshots/
      xxxxx/
        config.yaml
        ...
```

那就必须改成：

```python
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    cache_dir=r"models",
)
```

**请先确认 `models/` 目录的实际结构**，再决定用哪种写法。

### 2. 音频加载与 pipeline 调用

脚本用 `soundfile` 把音频读成了 `waveform`，但实际调用 pipeline 时传的是**文件路径**：

```python
output = pipeline(audio_path)
```

这意味着 `pyannote` 内部仍会走它自己的 IO 逻辑，也就是仍会碰 `torchcodec`。你前面用 `soundfile` 读的波形，只用于后面的音轨分离，没有用于 pipeline 分析本身。

如果 `torchcodec` 已经修好，这样写没问题；如果没修好，pipeline 调用这一步仍会报 `AudioDecoder` 相关的错。

### 3. `num_speakers` 没有真正传进去

脚本里定义了 `pipeline_kwargs`：

```python
pipeline_kwargs = {}
if num_speakers is not None:
    pipeline_kwargs["num_speakers"] = num_speakers

output = pipeline(audio_path)   # ← 这里没传 **pipeline_kwargs
```

`num_speakers` 参数被吞了，模型永远走自动检测。应改成：

```python
output = pipeline(audio_path, **pipeline_kwargs)
```

## 常见问题（FAQ）

### Q1: 运行时报 `torchcodec is not installed correctly so built-in audio decoding will fail`

**原因**：`torchcodec` 找不到 FFmpeg 的 DLL。

**解决**：
1. 确认下载的是 FFmpeg **shared** 版本（见上文）。
2. 在脚本最顶部调用 `os.add_dll_directory(FFMPEG_BIN)`。
3. 或把 FFmpeg `bin` 下的所有 DLL 复制到 `torchcodec` 安装目录。

### Q2: 运行时报 `NameError: name 'AudioDecoder' is not defined`

**原因**：这是 Q1 的连锁反应。`pyannote` 内部调用 `torchcodec.AudioDecoder`，但 `torchcodec` 因 DLL 加载失败而没有完成初始化，导致 `AudioDecoder` 这个名字未定义。

**解决**：按 Q1 修复 `torchcodec` 即可。修复前也可以临时用 `soundfile` 读取音频、把波形以字典形式传给 pipeline 来绕过：

```python
output = pipeline(
    {"waveform": waveform, "sample_rate": sample_rate},
    **pipeline_kwargs,
)
```

不过要注意：`community-1` 这条 pipeline 对纯内存字典的支持，不同版本行为不完全一致。如果传字典仍报 `AudioDecoder` 相关的错，说明这条 pipeline 没完全走内存路径，最终仍得靠 `os.add_dll_directory` 把 FFmpeg DLL 修好。

### Q3: `pip install pyannote.audio` 报版本找不到

**原因**：Python 版本过低。`pyannote.audio` 4.x 要求 Python ≥ 3.10。如果用的是 3.8 / 3.9，pip 会自动忽略 4.x 的版本。

**解决**：升级 Python，或改用 `pyannote.audio==3.1.1`（3.x 对 Python 版本要求较低，但 API 与 4.x 不同）。

### Q4: `pip` 报 `funasr 1.2.9 requires hydra-core>=1.3.2, but you have hydra-core 0.11.3`

**原因**：安装了 `denoiser` 之类的老库，它依赖 `hydra-core==0.11.3`，与 `funasr` 要求的 `>=1.3.2` 冲突。

**解决**：给 `denoiser` 单独建一个环境，不要和 `funasr`、`pyannote` 混在一起。

```bash
conda create -n denoiser_env python=3.8 -y
conda activate denoiser_env
pip install torch==1.11.0 torchaudio==0.11.0
pip install denoiser
```

### Q5: 模型加载报错，提示找不到 config 或权重文件

https://huggingface.co/pyannote/speaker-diarization-community-1

具体的模型调用方法

### Q6: `num_speakers` 传了没生效

**原因**：脚本里定义了 `pipeline_kwargs` 但调用时没展开。

**解决**：

```python
output = pipeline(audio_path, **pipeline_kwargs)
```


## 模型下载与离线使用

如果需要在无网络环境运行，可以提前把模型下载到本地。推荐使用 Hugging Face cache 结构，便于管理：

```python
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    cache_dir=r"models",   # 模型会缓存到 models/ 下
)
```

离线加载时，把 `models` 目录整体拷到目标机器，然后：

```python
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    cache_dir=r"models",
)
```

注意：如果 `models` 目录下已经是 `models--pyannote--xxx` 结构，用 `cache_dir` 方式；如果目录本身就是模型文件，则直接传目录路径给 `from_pretrained`。

## 如何使用

### 方式一：直接修改脚本内路径运行

当前脚本没有命令行参数，输入/输出路径和说话人数都写在 `if __name__ == "__main__":` 里：

```python
if __name__ == "__main__":
    INPUT_AUDIO = r"会议录音.wav"
    OUTPUT_DIR = r".\output"
    NUM_SPEAKERS = None
```

使用方法：

1. 把 `INPUT_AUDIO` 改成你的音频文件路径。
2. 把 `OUTPUT_DIR` 改成你想保存结果的目录。
3. 如果知道说话人数，把 `NUM_SPEAKERS` 改成对应整数；不确定就保持 `None`。
4. 运行：

```bash
python 4.0.4分离.py
```

**参数说明:**

*   `INPUT_AUDIO`: 输入的音频文件路径。推荐 `.wav` 格式。
*   `OUTPUT_DIR`: 保存分离后音频文件的目录。不存在时脚本会尝试创建。
*   `NUM_SPEAKERS`: 可选。整数，指定期望的说话人数，可帮助模型在语音重叠较多或声音相似时提高准确性。`None` 表示自动检测。

**示例:**

```python
# 处理 "会议录音.wav"，结果保存到 ".\output"
INPUT_AUDIO = r"会议录音.wav"
OUTPUT_DIR = r".\output"
NUM_SPEAKERS = None

# 明确告知模型有 2 位说话人
INPUT_AUDIO = r"my_interview.wav"
OUTPUT_DIR = r"output_files"
NUM_SPEAKERS = 2
```

## 注意事项

*   **处理时间**: 音频时长和硬件（CPU/GPU）会显著影响处理时间，长音频可能需要较长时间。
*   **准确性**: `pyannote.audio` 很强大，但在嘈杂环境、说话人声音相似或语音重叠严重时，分离结果可能不完美。
*   **内存消耗**: 处理非常长的音频可能消耗大量内存（RAM 和 GPU 显存）。
*   **音频格式**: 虽然脚本可能能处理 FFmpeg 支持的其他格式（如 MP3），但强烈建议输入预先转换为 `.wav` 格式（例如 16kHz 单声道 PCM）以获得最佳效果和兼容性。

## 致谢

本项目基于强大的 [pyannote.audio](https://github.com/pyannote/pyannote-audio) 库实现。感谢其开发者和社区。
