import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import modal

APP_NAME = "scail-ui"
MODEL_VOLUME_NAME = "scail-models"
MODEL_DIR = Path("/models/SCAIL-Preview")
REPO_DIR = Path("/root/app")
LOCAL_REPO_DIR = Path(__file__).parent

VIDEO_MODEL_CONFIG = "configs/video_model/Wan2.1-i2v-14Bsc-pose-xc-latent.yaml"
SAMPLING_BASE_CONFIG = "configs/sampling/wan_pose_14Bsc_xc_txt.yaml"

model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04", add_python="3.11"
    )
    .apt_install(
        "ffmpeg",
        "git",
        "libgl1",
        "libglib2.0-0",
        "libaio-dev",
        "build-essential",
    )
    .pip_install("ninja", "packaging", "wheel")
    .pip_install_from_requirements(
        "requirements.txt",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install("huggingface_hub", "hf_transfer")
    .pip_install("opencv-python==4.12.0.88", extra_options="--no-deps")
    .pip_install("numpy==1.26.4")
    .add_local_dir(
        LOCAL_REPO_DIR,
        remote_path=REPO_DIR.as_posix(),
        ignore=[".git", "__pycache__", "SCAIL-Preview"],
    )
)

app = modal.App(APP_NAME)


def _model_paths():
    return {
        "model_root": MODEL_DIR,
        "model_dir": MODEL_DIR / "model",
        "vae": MODEL_DIR / "Wan2.1_VAE.pth",
        "clip": MODEL_DIR
        / "models_clip_open-clip-xlm-roberta-large-vit-huge-14-onlyvisual.pth",
        "umt5_dir": MODEL_DIR / "umt5-xxl",
        "umt5_ckpt": MODEL_DIR / "umt5-xxl" / "models_t5_umt5-xxl-enc-bf16.pth",
    }


@app.function(
    image=image,
    volumes={"/models": model_volume},
    timeout=60 * 60 * 6,
)
def download_models():
    from huggingface_hub import snapshot_download

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER_COMPRESSION", "1")

    snapshot_download(
        repo_id="zai-org/SCAIL-Preview",
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    model_volume.commit()


def _assert_models_ready():
    missing = [p for p in _model_paths().values() if not p.exists()]
    if missing:
        missing_list = "\n".join(str(p) for p in missing)
        raise RuntimeError(
            "Model weights not found. Run: modal run modal_app.py::download_models\n"
            f"Missing:\n{missing_list}"
        )


def _write_video_model_config(dest_path: Path):
    from omegaconf import OmegaConf

    src_path = REPO_DIR / VIDEO_MODEL_CONFIG
    cfg = OmegaConf.load(src_path)
    paths = _model_paths()

    cfg.model.conditioner_config.params.emb_models[0].params.checkpoint_path = str(
        paths["umt5_ckpt"]
    )
    cfg.model.conditioner_config.params.emb_models[0].params.tokenizer_path = str(
        paths["umt5_dir"]
    )
    cfg.model.i2v_clip_config.params.checkpoint_path = str(paths["clip"])
    cfg.model.first_stage_config.params.vae_pth = str(paths["vae"])

    OmegaConf.save(cfg, dest_path)


def _write_sampling_config(dest_path: Path, input_file: Path, output_dir: Path):
    from omegaconf import OmegaConf

    src_path = REPO_DIR / SAMPLING_BASE_CONFIG
    cfg = OmegaConf.load(src_path)
    cfg.args.input_type = "txt"
    cfg.args.input_file = str(input_file)
    cfg.args.output_dir = str(output_dir)
    cfg.args.load = str(_model_paths()["model_dir"])
    OmegaConf.save(cfg, dest_path)


def _run_sample(prompt: str, ref_path: Path, pose_path: Path) -> Path:
    _assert_models_ready()

    run_id = uuid.uuid4().hex
    run_dir = Path(tempfile.mkdtemp(prefix="scail-"))
    input_dir = run_dir / "input"
    output_dir = run_dir / "outputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(ref_path, input_dir / "ref.jpg")
    shutil.copy(pose_path, input_dir / "rendered.mp4")

    input_file = run_dir / "input.txt"
    input_file.write_text(f"{prompt}@@{input_dir.as_posix()}\n", encoding="utf-8")

    video_model_cfg = run_dir / "video_model.yaml"
    sampling_cfg = run_dir / "sampling.yaml"
    _write_video_model_config(video_model_cfg)
    _write_sampling_config(sampling_cfg, input_file, output_dir)

    env = os.environ.copy()
    env.update(
        {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "LOCAL_WORLD_SIZE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "SCAIL_SKIP_DEEPSPEED": "1",
        }
    )

    cmd = [
        "python",
        "sample_video.py",
        "--base",
        str(video_model_cfg),
        str(sampling_cfg),
    ]
    subprocess.run(cmd, cwd=REPO_DIR, env=env, check=True)

    candidate_dir = output_dir / input_dir.name
    outputs = sorted(candidate_dir.glob("*_output_*.mp4"))
    if not outputs:
        raise RuntimeError("No output video found in output directory.")
    return outputs[-1]


def _coerce_path(value) -> Path:
    if isinstance(value, (tuple, list)) and value:
        return Path(value[0])
    if isinstance(value, dict) and "name" in value:
        return Path(value["name"])
    return Path(value)


@app.function(
    image=image,
    volumes={"/models": model_volume},
    gpu="A100-80GB",
    timeout=60 * 60,
    scaledown_window=60 * 10,
    max_containers=1,
)
@modal.concurrent(max_inputs=1)
@modal.web_server(port=7860, startup_timeout=60 * 30)
def serve():
    import gradio as gr

    def generate(prompt, ref_image, pose_video):
        if ref_image is None or pose_video is None:
            raise gr.Error("Please provide both a reference image and a pose video.")
        safe_prompt = prompt.strip() if prompt else "None"
        output_path = _run_sample(
            safe_prompt, _coerce_path(ref_image), _coerce_path(pose_video)
        )
        return str(output_path)

    with gr.Blocks(title="SCAIL - Studio-Grade Character Animation") as demo:
        gr.Markdown("# SCAIL - Studio-Grade Character Animation")
        gr.Markdown(
            "Upload a reference image and a pre-rendered pose video "
            "(rendered.mp4 from scail_pose)."
        )
        prompt = gr.Textbox(
            label="Prompt",
            placeholder="A woman with curly hair is joyfully dancing along a rocky shoreline...",
        )
        ref_image = gr.Image(label="Reference image", type="filepath")
        pose_video = gr.Video(label="Pose video (rendered.mp4)", format="mp4")
        run_btn = gr.Button("Generate")
        output_video = gr.Video(label="Output", format="mp4")

        run_btn.click(generate, inputs=[prompt, ref_image, pose_video], outputs=output_video)

    demo.queue(max_size=8)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        prevent_thread_lock=True,
        allowed_paths=[tempfile.gettempdir()],
    )
