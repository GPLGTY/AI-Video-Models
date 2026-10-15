# MiniMax-H3 Lite（H3L）部署说明（权威）

| 属性 | 内容 |
|------|------|
| 服务 | MiniMax-H3 轻量路径（ComfyUI pruned INT8 FL2VA/Ref2VA + NVFP4 TE；可选 GGUF / Turbo LoRA） |
| 版本 | 2.0 |
| 日期 | 2026-08-13 |
| 仓库 | `.` |
| ComfyUI | `third_party/ComfyUI`（**0.32.0+**，含原生 H3 节点） |
| 对照全量 | `optional external MiniMax-H3 installation`（SGLang BF16，端口 8960 / 30010） |
| 端口 | 平台网关 **8960**；ComfyUI **8962** |
| 8800 路由 | `POST /H3L`（别名 `/h3l`、`/h3lite`） |
| GPU | **GPU0 独占**（与全量 H3 互斥）；`CUDA_VISIBLE_DEVICES=0` |
| 关联规范 | `docs/AI推理服务部署规范.md` v1.3 §5/§2.7/§6 |

---

## 0. 与全量 H3 的关系

| 项 | 全量 MiniMax-H3 | 本服务 H3L |
|----|-----------------|------------|
| 运行栈 | SGLang + FastAPI | ComfyUI 0.32 + FastAPI |
| DiT | FL2VA/Ref2VA BF16 ~62GB | pruned INT8 convrot ~20GB |
| TE | BF16 ~63GB | NVFP4 AWQ ~15GB |
| 任务 ID 前缀 | `h3-` | `h3lite-` |
| 默认端口 | 8960 / 30010 | 8960 / 8962 |

**同一张 GPU0 不要同时跑全量 H3 与 H3L。** `scripts/start_h3lite_services.sh` 默认会停全量 H3；若需保留全量，设 `H3L_KEEP_FULL_H3=1`（可能 OOM）。

---

## 1. 后台启动方式（强制）

### 环境

- conda：`comfy-h3lite`（由 `minimax-h3` clone，含 `torch 2.13.0+cu130`）
- 激活：`source /opt/conda/etc/profile.d/conda.sh && conda activate comfy-h3lite`
- 依赖安装：`bash scripts/install_deps.sh`（已装过可跳过）
- 权重校验：`bash scripts/verify_lite_weights.sh`

### GPU

```bash
export CUDA_VISIBLE_DEVICES=0
```

### 公共环境 / OSS

```bash
source ./scripts/load_oss_env.sh
# 强制 OSS_KEY_PREFIX=h3l/jobs/
# MEDIA_DOWNLOAD_USE_PROXY 等见 zyxl-oss.sh
```

`OSS_ENABLED` 默认跟随 `/etc/profile.d/zyxl-oss.sh`（通常为 true）。已接入 `projects/shared_media`（§2.8）：排队预下载 + 共享资产缓存；推理成功后异步上传 OSS（上传期间对外仍 `running`）。上/下载合计最多 5 次；最终失败任务为 `failed`（不再“上传失败仍 finished”）。

此整理版默认关闭 本地结果保存；不要把任何对象存储密钥放进 Git。

### 前台（调试）

```bash
cd .
conda activate comfy-h3lite
source scripts/load_oss_env.sh
# 终端 1：ComfyUI
bash scripts/start_comfyui.sh
# 终端 2：平台网关
export H3L_COMFY_BASE=http://127.0.0.1:8962
python gateway_h3lite_fastapi.py --host 0.0.0.0 --port 8960
```

### 后台（推荐）

```bash
cd .
bash scripts/start_h3lite_services.sh
# PID: logs/h3l_comfyui.pid  logs/h3l_gateway.pid
```

### 停止 / 重启

```bash
bash ./scripts/stop_h3lite_services.sh
bash ./scripts/start_h3lite_services.sh
```

统一网关 8800 需加载已登记 H3L 的 `vectar/modelserver/method/gateway.py` 后重启。

### 权重

| 组件 | 路径 |
|------|------|
| DiT（FL2VA 默认） | `models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| DiT（Ref2VA） | `models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| TE（默认） | `models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| Video VAE | `models/vae/minimax_h3_video_vae_fp16.safetensors` |
| Audio VAE | `models/vae/minimax_h3_audio_vae_fp32.safetensors` |
| Turbo LoRA（可选） | `models/loras/minimax_h3_fl2v_turbo_4step_v0.1.safetensors` |
| GGUF 备选 | `models/unet/*.gguf` + TE Q4（需 ComfyUI-GGUF；默认路径未启用） |

ComfyUI 侧软链：`third_party/ComfyUI/models/...`（启动前已链接）。亦可 `bash scripts/link_into_comfyui.sh` 链到旧 FlashVSR 目录。

### 预热

- ComfyUI 启动后 `GET :8962/system_stats` 可通即认为 HTTP 就绪；**首次推理**会加载 ~35GB+ 权重，可能数分钟。
- 启动脚本默认 `H3L_ACCEPT_WHEN_UNHEALTHY=1`，加载期间可入队。

### 队列 / 超时（对齐平台 §5）

| 项 | 默认 |
|----|------|
| `H3L_MAX_PENDING_JOBS` | 16 |
| `H3L_JOB_TTL_SEC` | 14400（4h） |
| `H3L_JOB_TIMEOUT_SEC` | 10800（3h） |
| 进程内并发 | 1（串行跑 Comfy 任务） |
| 默认步数 | 25；`input.use_turbo=true` 时 4 + Turbo LoRA |
| 默认加速 | **H3-lite-Te**：工作流默认插入 `TESpeedMiniMaxH3`（`H3L_USE_TESPEED=1`）；请求可设 `input.use_tespeed=false` 回退 baseline |
| FL2VA 提示词重写 | **默认开启**（`mode=fl2va` 且有首帧）：经 Gemma（`H3L_GEMMA_BASE`，默认 `:8940`）按 `MiniMax-H3/skills/h3-prompt-writing` 重写后再推理。首+尾按 FL2VA；仅首帧按 I2VA。关闭：`H3L_FL2VA_PROMPT_REWRITE=0` 或请求 `input.prompt_rewrite=false` |

---

## 2. 周日志

| 项 | 约定 |
|----|------|
| 目录 | `MiniMax-H3-lite/logs/` |
| 网关 | `h3l_gateway_<周一>_to_<周日>.partNN.log`（实现文件名用 `_` 连接日期，与 §6.4 等价） |
| ComfyUI | `h3l_comfyui_<周一>_<周日>.partNN.log` |
| 周期 | 真实自然周 Mon–Sun，`Asia/Shanghai` |
| 检索 | 按 `h3lite-<uuid>` / `job=` / `prompt_id=` 检索 |

---

## 3. API（平台协议）

经统一网关：`POST http://<host>:8800/H3L`  
直连后端：`POST http://127.0.0.1:8960/H3L`

`input.duration_seconds` 有效范围 **[1, 15]**（默认 5）。越界时网关会**钳制**而非拒绝：`<1` → `1`，`>15` → `15`（日志 `duration_seconds clamped`）。短于约 5s 在模型训练常用区间之外，画质可能变差；实际帧数会对齐到 24fps 的 `17k+5` 网格（例如 `1` → 约 1.6s）。

### 提交（t2va）

```bash
curl -s http://127.0.0.1:8960/H3L -H 'Content-Type: application/json' -d '{
  "mode": "t2va",
  "input": {
    "prompt": "A calm ocean at sunset, cinematic, gentle waves, soft ambient sound.",
    "short_edge": 768,
    "aspect_ratio": "16:9",
    "duration_seconds": 5,
    "seed": 0
  }
}'
```

### 提交（fl2va / i2va）

`input.first_frame` / `input.last_frame`（或 `image`）支持 **HTTP(S) URL** 或本机路径。

分辨率策略（fl2va / i2va）：

- 有输入图时：**强制跟首张可用图的宽高比**（忽略客户端传入的固定 `16:9` 等），在 32 对齐与像素预算内取最接近比例。
- `short_edge`（如 `720`）表示输出**短边**目标；「720p」请传 `short_edge=720`，不是高度固定 720。
- ref2va 不变：尊重客户端 `aspect_ratio`；为 `adaptive` 时仍按**最后一张**参考图（通常是环境）估画幅。
- 无参考图的 t2va 仍默认 `aspect_ratio=16:9`。

```bash
curl -s http://127.0.0.1:8960/H3L -H 'Content-Type: application/json' -d '{
  "mode": "fl2va",
  "input": {
    "prompt": "The subject walks forward naturally, soft footsteps, ambient room tone.",
    "first_frame": "https://example.com/first.png",
    "last_frame": "https://example.com/last.png",
    "short_edge": 720,
    "aspect_ratio": "16:9",
    "duration_seconds": 5,
    "seed": 0
  }
}'
```

> 上例即使写了 `16:9`，竖版首帧仍会输出约 `704×1280` 竖屏（短边≈720）。

可选：`"use_turbo": true`（加载 Turbo LoRA，默认 4 steps）。

**首帧/首尾帧提示词重写（默认开）**：`fl2va` 且有 `first_frame`（或 `image`）时，网关会先调用 Gemma，按 `MiniMax-H3/skills/h3-prompt-writing` 重写 `prompt` 再送 Comfy——**仅首帧**按 I2VA；**首+尾帧**按 FL2VA。结果里含 `prompt_rewrite`（含 `variant=first_only|first_last`）/ `prompt_original` / `timing.prompt_rewrite_s`；前后全文落在 `outputs/<job_id>/prompt_before.txt` 与 `prompt_after.txt`。关闭：`"prompt_rewrite": false`。

### 提交（ref2va）

`input.ref_images` 为参考图列表（URL 或本地路径）；亦可用 `people` + `background`。  
另支持 **`ref_videos`（最多 3）**、**`ref_audios`（最多 3）**；可选 `ref_video_audios` 为视频配对音轨。  
默认 `use_video_soundtrack=true`：未指定配对音轨时自动使用视频自带音轨。  
提示词中用 `<Picture i>` / `<Video k>` / `<Audio j>` 指代（各类从 1 起算；呈现顺序：图 → 视频(+配对音) → 独立音频）。  
至少提供一类参考（图 / 视频 / 音频）。

```bash
curl -s http://127.0.0.1:8960/H3L -H 'Content-Type: application/json' -d '{
  "mode": "ref2va",
  "input": {
    "prompt": "Place the person from <Picture 1> into the scene from <Picture 2>...",
    "ref_images": ["/path/people.png", "/path/background.png"],
    "ref_videos": ["https://example.com/clip.mp4"],
    "ref_audios": ["https://example.com/sfx.wav"],
    "short_edge": 768,
    "aspect_ratio": "adaptive",
    "duration_seconds": 5,
    "seed": 0,
    "ref_image_size": "match",
    "use_video_soundtrack": true
  }
}'
```

> Ref2VA 使用独立 DiT（`minimax_h3_ref2va_pruned_int8_convrot.safetensors`），与 FL2VA 权重不同；同卡切换会触发重新加载。Turbo LoRA 仅适用于 FL2VA，ref2va 会忽略 `use_turbo`。  
> 参数说明 Word：`docs/H3L_Te_参考图模式_输入输出参数说明.docx`。

### 轮询（进行中）

```bash
curl -s http://127.0.0.1:8960/H3L -H 'Content-Type: application/json' \
  -d '{"id":"h3lite-<uuid>"}'
```

示例：

```json
{"id":"h3lite-...","code":0,"msg":"running","status":"running"}
```

### 成功终态（含 OSS）

```json
{
  "id": "h3lite-...",
  "code": 0,
  "msg": "finished",
  "status": "finished",
  "data": {
    "output_path": "./outputs/h3lite-.../xxx.mp4",
    "oss": {
      "objects": [
        {"local": "...", "key": "h3l/jobs/h3lite-.../xxx.mp4", "url": "https://..."}
      ],
      "job_id": "h3lite-..."
    }
  }
}
```

### OSS 3 次仍失败仍 finished

```json
{
  "status": "finished",
  "data": {
    "output_path": "/local/.../xxx.mp4",
    "oss": {"objects": [], "job_id": "h3lite-...", "error": "...(已重试 3 次)", "upload_retries": 3}
  }
}
```

### 失败示例

```json
{"id":"h3lite-...","code":50001,"msg":"failed","status":"failed","error":"RuntimeError: ComfyUI job failed: ..."}
```

### 健康检查

```bash
curl -s http://127.0.0.1:8960/health
curl -s http://127.0.0.1:8800/H3L/health
```

---

## 4. 依赖安装摘要

```bash
# 若尚无 env（本机已用 clone 建好）：
# conda create -n comfy-h3lite --clone minimax-h3 -y
cd .
# 若尚无 ComfyUI：
# git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git third_party/ComfyUI
bash scripts/install_deps.sh
bash scripts/verify_lite_weights.sh
```

INT8 convrot 需要 **CUDA 13 / torch cu130**（本 env 已满足）。

---

## 5. 变更记录

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0 | 2026-08-11 | 初版：仅落盘 Comfy 量化 / GGUF / Turbo；不启动 |
| 2.0 | 2026-08-13 | 平台化：ComfyUI 0.32 + `/H3L` 网关、周日志、URL/OSS、8800 登记；与全量 H3 互斥占 GPU0 |
| 2.1 | 2026-08-13 | 增加 Ref2VA：下载 pruned INT8 Ref2VA DiT；网关 `mode=ref2va` + `MiniMaxH3ReferenceToVideo` |
