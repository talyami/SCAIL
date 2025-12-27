import json
import os
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "wan2gp-scail"
MODEL_VOLUME_NAME = "wan2gp-models"
OUTPUT_VOLUME_NAME = "wan2gp-outputs"

MODEL_DIR = Path("/models")
OUTPUT_DIR = Path("/outputs")
CONFIG_DIR = MODEL_DIR / "config"
REPO_DIR = Path("/root/app")
LOCAL_REPO_DIR = Path(__file__).parent

WAN_REPO = "DeepBeepMeep/Wan2.1"

SCAIL_DOWNLOADS = {
    "": [
        "wan2.1_scail_preview_14B_quanto_bf16_int8.safetensors",
        "Wan2.1_VAE.safetensors",
        "Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors",
        "fantasy_proj_model.safetensors",
    ],
    "xlm-roberta-large": [
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14-bf16.safetensors",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ],
    "umt5-xxl": [
        "models_t5_umt5-xxl-enc-quanto_int8.safetensors",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer_config.json",
    ],
    "pose": [
        "dw-ll_ucoco_384.onnx",
        "yolox_l.onnx",
        "nlf_l_multi_0.3.2.eager.safetensors",
        "nlf_l_multi_0.3.2.eager.meta.json",
    ],
    "mask": [
        "sam_vit_h_4b8939_fp16.safetensors",
        "model.safetensors",
        "config.json",
    ],
}


model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install(
        "ffmpeg",
        "git",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
        "build-essential",
        "cmake",
        "ninja-build",
    )
    .pip_install("pip", "setuptools", "wheel")
    .pip_install(
        "torch==2.6.0+cu124",
        "torchvision==0.21.0+cu124",
        "torchaudio==2.6.0+cu124",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install_from_requirements("requirements.modal.txt")
    .pip_install("hf_transfer", "huggingface_hub", "jupyter")
    .add_local_dir(
        LOCAL_REPO_DIR,
        remote_path=REPO_DIR.as_posix(),
        ignore=[".git", "__pycache__", "ckpts", "outputs"],
    )
)

app = modal.App(APP_NAME)


def _setup_hf_env():
    cache_dir = MODEL_DIR / "hf"
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER_COMPRESSION", "1")


def _ensure_wgp_config():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = CONFIG_DIR / "wgp_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        from mmgp import profile_type

        config = {
            "attention_mode": "auto",
            "transformer_types": [],
            "transformer_quantization": "int8",
            "text_encoder_quantization": "int8",
            "save_path": str(OUTPUT_DIR),
            "image_save_path": str(OUTPUT_DIR),
            "compile": "",
            "metadata_type": "metadata",
            "boost": 1,
            "clear_file_list": 5,
            "vae_config": 0,
            "profile": profile_type.LowRAM_LowVRAM,
            "preload_model_policy": [],
            "UI_theme": "default",
            "checkpoints_paths": [str(MODEL_DIR)],
            "queue_color_scheme": "pastel",
            "model_hierarchy_type": 1,
        }

    config["checkpoints_paths"] = [str(MODEL_DIR)]
    config["save_path"] = str(OUTPUT_DIR)
    config["image_save_path"] = str(OUTPUT_DIR)
    config["last_model_type"] = "scail"
    config.setdefault("transformer_quantization", "int8")
    config.setdefault("text_encoder_quantization", "int8")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _download_repo_files(repo_id: str, subfolder: str, filenames: list[str]):
    from huggingface_hub import hf_hub_download

    for filename in filenames:
        target = MODEL_DIR / subfolder / filename if subfolder else MODEL_DIR / filename
        if target.exists():
            continue
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder or None,
            local_dir=str(MODEL_DIR),
            local_dir_use_symlinks=False,
            resume_download=True,
        )


@app.function(
    image=image,
    volumes={MODEL_DIR.as_posix(): model_volume, OUTPUT_DIR.as_posix(): output_volume},
    timeout=60 * 60 * 6,
)
def download_scail_assets():
    _setup_hf_env()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_wgp_config()

    for subfolder, filenames in SCAIL_DOWNLOADS.items():
        _download_repo_files(WAN_REPO, subfolder, filenames)

    model_volume.commit()


@app.function(
    image=image,
    volumes={MODEL_DIR.as_posix(): model_volume, OUTPUT_DIR.as_posix(): output_volume},
    gpu="A10G",
    timeout=60 * 60 * 6,
    scaledown_window=60 * 60,
    min_containers=1,
    max_containers=1,
)
@modal.concurrent(max_inputs=1)
@modal.web_server(port=7860, startup_timeout=60 * 30)
def serve():
    _setup_hf_env()
    _ensure_wgp_config()

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WAN2GP_SKIP_PLUGIN_AUTOINSTALL"] = "1"

    cmd = [
        "python",
        "wgp.py",
        "--listen",
        "--server-port",
        "7860",
        "--config",
        str(CONFIG_DIR),
    ]
    subprocess.Popen(cmd, cwd=REPO_DIR, env=env).wait()


@app.function(
    image=image,
    volumes={MODEL_DIR.as_posix(): model_volume, OUTPUT_DIR.as_posix(): output_volume},
    gpu="A10G",
    timeout=60 * 60 * 6,
    max_containers=1,
)
def serve_tunnel(timeout: int = 60 * 60 * 6):
    _setup_hf_env()
    _ensure_wgp_config()

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WAN2GP_SKIP_PLUGIN_AUTOINSTALL"] = "1"

    cmd = [
        "python",
        "wgp.py",
        "--listen",
        "--server-port",
        "7860",
        "--config",
        str(CONFIG_DIR),
    ]

    with modal.forward(7860) as tunnel:
        proc = subprocess.Popen(cmd, cwd=REPO_DIR, env=env)
        print(f"WanGP UI (tunnel): {tunnel.url}")
        try:
            end_time = time.time() + timeout
            while time.time() < end_time:
                time.sleep(5)
        finally:
            proc.terminate()


@app.function(
    image=image,
    volumes={MODEL_DIR.as_posix(): model_volume, OUTPUT_DIR.as_posix(): output_volume},
    gpu="A10G",
    timeout=60 * 60 * 6,
    max_containers=1,
)
def notebook(timeout: int = 60 * 60 * 6):
    _setup_hf_env()
    _ensure_wgp_config()

    import secrets

    token = os.environ.get("WAN2GP_JUPYTER_TOKEN") or secrets.token_urlsafe(16)
    jupyter_port = 8888
    env = {
        **os.environ,
        "JUPYTER_TOKEN": token,
        "TOKENIZERS_PARALLELISM": "false",
    }

    with modal.forward(jupyter_port) as tunnel:
        proc = subprocess.Popen(
            [
                "jupyter",
                "notebook",
                "--no-browser",
                "--allow-root",
                "--ip=0.0.0.0",
                f"--port={jupyter_port}",
                "--NotebookApp.allow_origin=*",
                "--NotebookApp.allow_remote_access=1",
                f"--NotebookApp.notebook_dir={REPO_DIR.as_posix()}",
            ],
            env=env,
        )
        print(f"Jupyter available at {tunnel.url} (token: {token})")

        try:
            end_time = time.time() + timeout
            while time.time() < end_time:
                time.sleep(5)
        finally:
            proc.kill()
