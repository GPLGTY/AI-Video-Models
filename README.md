# AI Video Models: LatentSync、SoulX-FlashTalk 与 H3-lite-Te

本仓库将三个视频生成/口型同步项目集中整理。源码与启动脚本在仓库内；**模型权重不纳入 Git**，请按各项目下文给出的官方地址下载。模型许可与原始项目许可分别适用。GPU/CUDA 版本及显存要求会影响安装；先看每个子项目说明。

> H3-lite-Te 基于本机部署代码整理为独立示例：移除了私有 OSS/shared-media 依赖与机器绝对路径；默认只在本地运行，OSS 上传关闭。TE-Speed 是 ComfyUI 工作流节点/加速组件，不是额外的大模型权重。请按其上游节点提供的安装和许可要求单独安装该节点。

## 目录

- `LatentSync/`：ByteDance LatentSync 1.6，唇形同步。
- `FlashTalk/`：SoulX-FlashTalk，音频驱动头像视频。
- `H3-lite-Te/`：MiniMax-H3 ComfyUI 轻量路径与 TE-Speed 工作流。

## 快速开始

建议 Linux + NVIDIA GPU，分别创建隔离环境，不要把三个项目的依赖装进同一个 Python 环境。

### LatentSync

```bash
cd LatentSync
# 上游脚本创建 latentsync conda 环境、安装 requirements，并下载默认权重
bash setup_env.sh
conda activate latentsync
# Gradio UI
python gradio_app.py
# 或 CLI（按 inference.sh 设置输入视频/音频路径）
bash inference.sh
```

默认权重：`ByteDance/LatentSync-1.6`，下载地址 [Hugging Face 模型仓库](https://huggingface.co/ByteDance/LatentSync-1.6)。脚本下载 `latentsync_unet.pt` 与 `whisper/tiny.pt` 到 `LatentSync/checkpoints/`。官方代码说明最低显存约 18 GB（1.6）；具体结果受分辨率与推理设置影响。

### SoulX-FlashTalk

```bash
cd FlashTalk
conda create -n flashtalk python=3.10 -y
conda activate flashtalk
# 按官方 Quickstart 安装对应 CUDA 的 PyTorch（README 示例为 CUDA 12.8）
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install ninja
pip install flash_attn==2.8.0.post2 --no-build-isolation
conda install -c conda-forge ffmpeg -y
# 下载权重到推理脚本预期的目录（首次下载约数十 GB）
pip install -U huggingface_hub
huggingface-cli download Soul-AILab/SoulX-FlashTalk-14B --local-dir ./models/SoulX-FlashTalk-14B
huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./models/chinese-wav2vec2-base
bash inference_script_single_gpu.sh
```

权重：
- 14B 主模型：[Soul-AILab/SoulX-FlashTalk-14B](https://huggingface.co/Soul-AILab/SoulX-FlashTalk-14B)
- 中文音频编码器：[TencentGameMate/chinese-wav2vec2-base](https://huggingface.co/TencentGameMate/chinese-wav2vec2-base)

官方单卡说明要求超过 64 GB 显存；启用 `--cpu_offload` 可降低到约 40 GB，仍需大显存。多卡脚本与具体模型配置见 `FlashTalk/README.md`。

### H3-lite-Te

详细条件、权重列表与 API 示例见 [`H3-lite-Te/docs/DEPLOY.md`](H3-lite-Te/docs/DEPLOY.md)。该路径需要兼容的 ComfyUI MiniMax-H3 节点、NVIDIA GPU，以及与 INT8/量化实现匹配的 PyTorch/CUDA。

```bash
cd H3-lite-Te
# 可选：设置 CUDA_VISIBLE_DEVICES、TORCH_INDEX_URL、H3L_CONDA_ENV
bash scripts/install_deps.sh
bash scripts/download_lite_weights.sh
bash scripts/verify_lite_weights.sh
bash scripts/start_h3lite_services.sh
curl http://127.0.0.1:8960/health
# 停止只会向本仓库启动的 PID 发信号，不会按端口杀死其它进程
bash scripts/stop_h3lite_services.sh
```

H3-lite 权重下载地址与模型 ID：
- FL2VA/Ref2VA 量化 DiT、NVFP4 AWQ 文本编码器及 VAE：[Comfy-Org/MiniMax-H3（ModelScope）](https://modelscope.cn/models/Comfy-Org/MiniMax-H3)
- 可选 GGUF：[realrebelai/MiniMax-H3_GGUFs（ModelScope）](https://modelscope.cn/models/realrebelai/MiniMax-H3_GGUFs)
- 可选 Turbo LoRA：[lightx2v/Minimax-h3-Turbo（Hugging Face）](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
- H3-lite-Te/TE-Speed 是加速节点而非额外 checkpoint；需要在 ComfyUI 中按该节点上游的说明另行安装。

默认下载脚本下载模型仓库权重，不会下载大型权重到 Git，也不会配置任何个人对象存储凭据。若模型仓库要求登录/接受许可，请先在对应平台完成授权。

## Git 与许可证

每个项目保留其上游许可证/版权声明（若上游提供）。本仓库不将上游代码重新许可；使用前请自行核对模型权重、第三方 ComfyUI 节点以及各权重卡的使用条款。
