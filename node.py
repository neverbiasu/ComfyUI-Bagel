import os
import sys
import json
import random
import numpy as np
import torch
import subprocess
import traceback
from typing import Dict, Tuple, Optional, Any, Union
from PIL import Image
from folder_paths import folder_names_and_paths, models_dir as comfy_models_dir
from huggingface_hub import snapshot_download

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import BAGEL related modules
try:
    from accelerate import (
        infer_auto_device_map,
        load_checkpoint_and_dispatch,
        init_empty_weights,
        dispatch_model,  # Added for DFloat11 multi-GPU
    )
    from data.data_utils import add_special_tokens, pil_img2rgb
    from data.transforms import ImageTransform
    from inferencer import InterleaveInferencer
    from modeling.autoencoder import load_ae
    from modeling.bagel.qwen2_navit import NaiveCache
    from modeling.bagel import (
        BagelConfig,
        Bagel,
        Qwen2Config,
        Qwen2ForCausalLM,
        SiglipVisionConfig,
        SiglipVisionModel,
    )
    from modeling.qwen2 import Qwen2Tokenizer
except ImportError as e:
    print(f"Error importing BAGEL modules: {e}")
    print("Please ensure BAGEL model files are properly installed.")

try:
    from dfloat11 import DFloat11Model
except ImportError:
    print("DFloat11Model not found. DFloat11 support will be unavailable.")
    print(
        "Please install DFloat11 if you intend to use DFloat11 models: pip install dfloat11"
    )
    DFloat11Model = None

# Register the BAGEL model folder
folder_names_and_paths["bagel"] = (
    [os.path.join(comfy_models_dir, "bagel")],
    [".json", ".safetensors"],
)
# Register the VAE model folder
folder_names_and_paths["vae"] = (
    [os.path.join(comfy_models_dir, "vae")],
    [".safetensors"],
)


def set_seed(seed: int) -> int:
    """Set random seeds for reproducibility"""
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def download_model_with_git(
    model_dir: str, repo_id: str = "ByteDance-Seed/BAGEL-7B-MoT"
) -> str:
    """
    Download model using git lfs (recommended method)

    Args:
        model_dir: Directory to download the repo to (repo files will be placed directly here)
        repo_id: Hugging Face repository ID

    Returns:
        Path to the downloaded model if successful, None otherwise
    """
    try:
        print(f"Downloading BAGEL model using git lfs to {model_dir}...")

        # Create parent directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Check if git lfs is installed
        try:
            subprocess.run(["git", "lfs", "version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Git LFS not found. Installing git lfs...")
            subprocess.run(["git", "lfs", "install"], check=True)

        # Clone the repository directly to model_dir
        clone_cmd = ["git", "clone", f"https://huggingface.co/{repo_id}", model_dir]

        result = subprocess.run(clone_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Successfully downloaded BAGEL model to {model_dir}")
            return model_dir
        else:
            print(f"Git clone failed: {result.stderr}")
            return None

    except Exception as e:
        print(f"Error downloading model with git: {e}")
        return None


def download_model_with_hf_hub(
    model_dir: str, repo_id: str = "ByteDance-Seed/BAGEL-7B-MoT"
) -> str:
    """
    Download model using huggingface_hub (fallback method)

    Args:
        model_dir: Directory to download the repo to (repo files will be placed directly here)
        repo_id: Hugging Face repository ID

    Returns:
        Path to the downloaded model if successful, None otherwise
    """
    try:
        from huggingface_hub import snapshot_download

        print(f"Downloading BAGEL model using huggingface_hub to {model_dir}...")

        # Create parent directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Download the entire repository directly to model_dir
        snapshot_download(
            repo_id=repo_id, local_dir=model_dir, local_dir_use_symlinks=False
        )

        print(f"Successfully downloaded BAGEL model to {model_dir}")
        return model_dir

    except ImportError:
        print(
            "huggingface_hub not installed. Please install it with: pip install huggingface_hub"
        )
        return None
    except Exception as e:
        print(f"Error downloading model with huggingface_hub: {e}")
        return None


def check_model_files(model_path: str, is_df11_model: bool) -> bool:
    """
    Check if core model configuration files exist.
    VAE and main weights (ema.safetensors for standard) are checked separately during load.

    Args:
        model_path: Path to the model directory
        is_df11_model: Boolean indicating if the model is DFloat11

    Returns:
        True if core config files exist, False otherwise
    """
    required_files = [
        "llm_config.json",
        "vit_config.json",
    ]
    # DFloat11 models do not have ema.safetensors in their root.
    # Standard models expect ema.safetensors.
    # VAE presence is checked more robustly during the loading process itself.
    if not is_df11_model:
        required_files.append("ema.safetensors")

    for file_name in required_files:
        if not os.path.exists(os.path.join(model_path, file_name)):
            print(f"Missing required model file: {os.path.join(model_path, file_name)}")
            return False
    return True


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to ComfyUI tensor format"""
    img_array = np.array(img).astype(np.float32) / 255.0
    if len(img_array.shape) == 3:
        img_tensor = torch.from_numpy(img_array)[None,]  # Add batch dimension
    else:
        img_tensor = torch.from_numpy(img_array)
    return img_tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI tensor to PIL image"""
    if len(tensor.shape) == 4:
        tensor = tensor[0]  # Remove batch dimension
    img_array = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_array)


class BagelModelLoader:
    """BAGEL Model Loader Node"""

    SUPPORTED_MODEL_REPOS = [
        "ByteDance-Seed/BAGEL-7B-MoT",  # Standard BAGEL
        "DFloat11/BAGEL-7B-MoT-DF11",  # DFloat11 Quantized
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_repo_id": (
                    cls.SUPPORTED_MODEL_REPOS,
                    {"default": cls.SUPPORTED_MODEL_REPOS[0]},
                ),
            }
        }

    RETURN_TYPES = ("BAGEL_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "BAGEL/Core"

    @classmethod
    def VALIDATE_INPUTS(cls, model_repo_id):
        """Validate input parameters"""
        if model_repo_id not in cls.SUPPORTED_MODEL_REPOS:
            return f"Unsupported model_repo_id: {model_repo_id}. Supported: {cls.SUPPORTED_MODEL_REPOS}"

        if "DFloat11" in model_repo_id and DFloat11Model is None:
            return "DFloat11 model selected, but DFloat11Model library is not installed or failed to import. Please install it: pip install dfloat11"

        return True

    def load_model(self, model_repo_id: str) -> Tuple[Dict[str, Any]]:
        """
        Load BAGEL model and its components. Supports standard and DFloat11 models.
        Automatically downloads the model if not found.

        Args:
            model_repo_id: Hugging Face model repository ID (e.g., "ByteDance-Seed/BAGEL-7B-MoT" or "DFloat11/BAGEL-7B-MoT-DF11")

        Returns:
            Dictionary containing all model components
        """
        try:
            is_df11_model = "DFloat11" in model_repo_id
            print(f"Loading BAGEL model: {model_repo_id} (DFloat11: {is_df11_model})")

            base_repo_dir = os.path.join(comfy_models_dir, "bagel")
            repo_name_segment = model_repo_id.split("/")[-1]
            local_model_dir = os.path.join(base_repo_dir, repo_name_segment)

            common_vae_dir = os.path.join(comfy_models_dir, "vae")
            common_vae_file = os.path.join(common_vae_dir, "ae.safetensors")
            os.makedirs(common_vae_dir, exist_ok=True)
            os.makedirs(local_model_dir, exist_ok=True)

            if not check_model_files(local_model_dir, is_df11_model):
                print(
                    f"Core model files not found or incomplete for {model_repo_id} at {local_model_dir}. Attempting download..."
                )

                download_successful = False
                if is_df11_model:
                    if download_model_with_git(local_model_dir, repo_id=model_repo_id):
                        download_successful = True
                    else:
                        print(
                            "Git download failed, falling back to huggingface_hub for DFloat11 model..."
                        )
                        if download_model_with_hf_hub(
                            local_model_dir, repo_id=model_repo_id
                        ):
                            download_successful = True
                else:
                    if download_model_with_hf_hub(
                        local_model_dir, repo_id=model_repo_id
                    ):
                        download_successful = True

                if not download_successful:
                    raise FileNotFoundError(
                        f"Failed to download model {model_repo_id}. Please manually download it and place it in {local_model_dir}"
                    )
                print(
                    f"Successfully downloaded model {model_repo_id} to {local_model_dir}"
                )

            if not check_model_files(
                local_model_dir, is_df11_model
            ):  # Re-check after download
                raise FileNotFoundError(
                    f"Required model configuration files missing in {local_model_dir} for {model_repo_id} even after download attempt."
                )

            llm_config_path = os.path.join(local_model_dir, "llm_config.json")
            vit_config_path = os.path.join(local_model_dir, "vit_config.json")

            llm_config = Qwen2Config.from_json_file(llm_config_path)
            llm_config.qk_norm = True
            llm_config.tie_word_embeddings = False
            llm_config.layer_module = "Qwen2MoTDecoderLayer"

            vit_config = SiglipVisionConfig.from_json_file(vit_config_path)
            vit_config.rope = False
            vit_config.num_hidden_layers -= 1

            vae_model, vae_config = None, None
            potential_vae_paths = []
            if is_df11_model:
                potential_vae_paths.append(
                    os.path.join(local_model_dir, "vae", "ae.safetensors")
                )
            potential_vae_paths.append(os.path.join(local_model_dir, "ae.safetensors"))
            potential_vae_paths.append(common_vae_file)

            vae_loaded_path = None
            for vae_path_to_try in potential_vae_paths:
                if os.path.exists(vae_path_to_try):
                    try:
                        vae_model, vae_config = load_ae(local_path=vae_path_to_try)
                        if vae_model is not None and vae_config is not None:
                            vae_loaded_path = vae_path_to_try
                            break
                    except Exception as e:
                        print(f"Failed to load VAE from {vae_path_to_try}: {e}")

            if vae_loaded_path:
                print(f"Successfully loaded VAE from: {vae_loaded_path}")
            else:
                raise FileNotFoundError(
                    f"VAE model (ae.safetensors) could not be loaded from any of the expected paths: {potential_vae_paths}"
                )

            config = BagelConfig(
                visual_gen=True,
                visual_und=True,
                llm_config=llm_config,
                vit_config=vit_config,
                vae_config=vae_config,
                vit_max_num_patch_per_side=70,
                connector_act="gelu_pytorch_tanh",
                latent_patch_size=2,
                max_latent_size=64,
            )

            with init_empty_weights():
                language_model = Qwen2ForCausalLM(llm_config)
                vit_model = SiglipVisionModel(vit_config)
                model = Bagel(language_model, vit_model, config)
                model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
                    vit_config, meta=True
                )

            tokenizer = Qwen2Tokenizer.from_pretrained(local_model_dir)
            tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
            vae_transform = ImageTransform(1024, 512, 16)
            vit_transform = ImageTransform(980, 224, 14)
            model_device_final_str = "cpu"

            if is_df11_model:
                if DFloat11Model is None:
                    raise RuntimeError(
                        "DFloat11Model is not available. Please install DFloat11 library."
                    )
                model = model.to(torch.bfloat16)
                cpu_state_dict = {
                    n: torch.empty(p.shape, dtype=p.dtype, device="cpu")
                    for n, p in model.named_parameters()
                }
                model.load_state_dict(cpu_state_dict, assign=True)
                model = DFloat11Model.from_pretrained(
                    local_model_dir, bfloat16_model=model, device="cpu"
                )

                num_gpus = torch.cuda.device_count()
                if num_gpus > 0:
                    target_device_str = "cuda:0" if num_gpus == 1 else "auto"
                    if num_gpus > 1:
                        print(f"Dispatching DFloat11 model to {num_gpus} GPUs...")
                        no_split_module_classes = [
                            "Bagel",
                            "Qwen2MoTDecoderLayer",
                            "SiglipVisionModel",
                        ]
                        max_memory = {
                            i: f"{torch.cuda.get_device_properties(i).total_memory // (1024**3) - 2}GiB"
                            for i in range(num_gpus)
                        }
                        device_map = infer_auto_device_map(
                            model,
                            no_split_module_classes=no_split_module_classes,
                            max_memory=max_memory,
                            dtype=torch.bfloat16,
                        )

                        lm_head_device = device_map.get("language_model.lm_head")
                        if lm_head_device:
                            for module_key_prefix in ["vit_model", "connector"]:
                                for actual_module_name in device_map.keys():
                                    if actual_module_name.startswith(module_key_prefix):
                                        if (
                                            device_map[actual_module_name]
                                            != lm_head_device
                                        ):
                                            device_map[actual_module_name] = (
                                                lm_head_device
                                            )
                                        break
                                else:
                                    pass

                        dispatch_model(model, device_map=device_map)
                        model_device_final_str = "cuda (multi-gpu)"
                    else:  # Single GPU
                        no_split_module_classes_single_gpu = [
                            "Qwen2MoTDecoderLayer",
                            "SiglipVisionModel",
                        ]
                        gpu_index = int(target_device_str.split(":")[-1])
                        total_gpu_memory_gb = torch.cuda.get_device_properties(
                            gpu_index
                        ).total_memory / (1024**3)
                        headroom_gb = (
                            min(2, total_gpu_memory_gb - 1)
                            if total_gpu_memory_gb > 1
                            else 0.5
                        )
                        usable_memory_gb = total_gpu_memory_gb - headroom_gb
                        if usable_memory_gb < 1:
                            usable_memory_gb = total_gpu_memory_gb * 0.8

                        max_memory_single_gpu = {
                            gpu_index: f"{usable_memory_gb:.1f}GiB"
                        }

                        device_map = infer_auto_device_map(
                            model,
                            no_split_module_classes=no_split_module_classes_single_gpu,
                            max_memory=max_memory_single_gpu,
                            dtype=torch.bfloat16,
                            clean_result=False,
                        )

                        lm_head_device_actual = device_map.get("language_model.lm_head")
                        if lm_head_device_actual is None:
                            lm_block_device = device_map.get("language_model")
                            if lm_block_device is not None:
                                lm_head_device_actual = lm_block_device

                        if lm_head_device_actual is not None:
                            for module_key_prefix in ["vit_model", "connector"]:
                                for actual_module_name in device_map.keys():
                                    if actual_module_name.startswith(module_key_prefix):
                                        if (
                                            device_map[actual_module_name]
                                            != lm_head_device_actual
                                        ):
                                            device_map[actual_module_name] = (
                                                lm_head_device_actual
                                            )
                                        break
                                else:
                                    pass

                        dispatch_model(model, device_map=device_map)
                        try:
                            final_device_candidate = None
                            if lm_head_device_actual is not None:
                                final_device_candidate = lm_head_device_actual
                            elif (
                                "language_model.embed_tokens" in device_map
                                and device_map.get("language_model.embed_tokens")
                                is not None
                            ):
                                final_device_candidate = device_map[
                                    "language_model.embed_tokens"
                                ]
                            elif "" in device_map and device_map.get("") is not None:
                                final_device_candidate = device_map[""]

                            if final_device_candidate is not None:
                                if isinstance(final_device_candidate, int):
                                    model_device_final_str = (
                                        f"cuda:{final_device_candidate}"
                                    )
                                elif isinstance(final_device_candidate, str):
                                    if final_device_candidate.isdigit():
                                        model_device_final_str = (
                                            f"cuda:{final_device_candidate}"
                                        )
                                    else:
                                        model_device_final_str = final_device_candidate
                                else:
                                    print(
                                        f"Warning: Unexpected device type '{type(final_device_candidate)}' for final_device_candidate. Falling back to target_device_str."
                                    )
                                    model_device_final_str = target_device_str
                            else:
                                print(
                                    "Warning: Could not determine specific device from device_map. Falling back to target_device_str."
                                )
                                model_device_final_str = target_device_str
                        except Exception as e_dev_query:
                            print(
                                f"Exception querying device after dispatch: {e_dev_query}"
                            )
                            model_device_final_str = target_device_str

                else:  # CPU only
                    model_device_final_str = "cpu"
                    print("DFloat11 model remains on CPU (no CUDA devices found).")
                model = model.eval()
            else:  # Standard BAGEL model loading
                model_checkpoint_path = os.path.join(local_model_dir, "ema.safetensors")
                if not os.path.exists(model_checkpoint_path):
                    raise FileNotFoundError(
                        f"Standard model weights (ema.safetensors) not found at {model_checkpoint_path}"
                    )
                model = load_checkpoint_and_dispatch(
                    model,
                    checkpoint=model_checkpoint_path,
                    device_map="auto",
                    dtype=torch.bfloat16,
                    force_hooks=True,
                    offload_buffers=True,
                ).eval()
                try:
                    model_device_final_str = str(next(model.parameters()).device)
                except StopIteration:
                    model_device_final_str = (
                        "cpu"  # Should not happen for a loaded model
                    )

            vae_target_device_str_cleaned = model_device_final_str.split(" ")[0]
            if (
                "cuda" in vae_target_device_str_cleaned
                and not torch.cuda.is_available()
            ):
                vae_target_device_str_cleaned = "cpu"
            if (  # Ensure "cuda" becomes "cuda:0" if no index specified
                ":" not in vae_target_device_str_cleaned
                and "cuda" in vae_target_device_str_cleaned
            ):
                vae_target_device_str_cleaned = "cuda:0"

            vae_target_device = torch.device(vae_target_device_str_cleaned)
            try:
                current_vae_device = next(vae_model.parameters()).device
                if current_vae_device != vae_target_device:
                    print(
                        f"Moving VAE model from {current_vae_device} to device: {vae_target_device}"
                    )
                    vae_model = vae_model.to(vae_target_device)
            except StopIteration:
                print(
                    f"Warning: VAE model of type {type(vae_model).__name__} has no parameters. Attempting to move to {vae_target_device} regardless."
                )
                vae_model = vae_model.to(vae_target_device)
            except AttributeError as e:
                print(
                    f"Error accessing parameters of VAE model: {e}. VAE type: {type(vae_model).__name__}. Attempting to move to {vae_target_device} regardless."
                )
                try:
                    vae_model = vae_model.to(vae_target_device)
                except Exception as move_err:
                    print(
                        f"Failed to move VAE model after parameter access error: {move_err}"
                    )
                    raise RuntimeError(
                        f"VAE model ({type(vae_model).__name__}) could not be reliably moved to device after an issue determining its current device."
                    ) from e

            inferencer = InterleaveInferencer(
                model=model,
                vae_model=vae_model,
                tokenizer=tokenizer,
                vae_transform=vae_transform,
                vit_transform=vit_transform,
                new_token_ids=new_token_ids,
            )

            model_dict = {
                "model": model,
                "inferencer": inferencer,
                "tokenizer": tokenizer,
                "vae_model": vae_model,
                "vae_transform": vae_transform,
                "vit_transform": vit_transform,
                "config": config,
                "model_path": local_model_dir,
                "model_repo_id": model_repo_id,
                "is_df11": is_df11_model,
                "device": model_device_final_str,  # This is the main device string for the Bagel model
            }
            print(
                f"Successfully loaded BAGEL model '{model_repo_id}'. Final model primary device: {model_device_final_str}"
            )
            return (model_dict,)

        except Exception as e:
            print(f"Unhandled error loading BAGEL model '{model_repo_id}': {e}")
            traceback.print_exc()
            raise e


class BagelTextToImage:
    """BAGEL Text to Image Node"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BAGEL_MODEL", {"tooltip": "BAGEL model"}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A female cosplayer portraying an ethereal fairy or elf, wearing a flowing dress made of delicate fabrics in soft, mystical colors like emerald green and silver. She has pointed ears, a gentle, enchanting expression, and her outfit is adorned with sparkling jewels and intricate patterns. The background is a magical forest with glowing plants, mystical creatures, and a serene atmosphere.",
                        "tooltip": "Text prompt",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000000,
                        "tooltip": "Random seed, 0 for random",
                    },
                ),
                "image_ratio": (
                    ["1:1", "4:3", "3:4", "16:9", "9:16"],
                    {"default": "1:1", "tooltip": "Image aspect ratio"},
                ),
                "cfg_text_scale": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 1.0,
                        "max": 8.0,
                        "step": 0.1,
                        "tooltip": "CFG text scaling",
                    },
                ),
                "num_timesteps": (
                    "INT",
                    {
                        "default": 50,
                        "min": 10,
                        "max": 100,
                        "step": 5,
                        "tooltip": "Denoising steps",
                    },
                ),
            },
            "optional": {
                "show_thinking": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Display reasoning process"},
                ),
                "cfg_interval": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": "CFG interval start value",
                    },
                ),
                "timestep_shift": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 1.0,
                        "max": 5.0,
                        "step": 0.5,
                        "tooltip": "Timestep offset",
                    },
                ),
                "cfg_renorm_min": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": "CFG re-normalization minimum value",
                    },
                ),
                "cfg_renorm_type": (
                    ["global", "local", "text_channel"],
                    {"default": "global", "tooltip": "CFG re-normalization type"},
                ),
                "text_temperature": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": "Text generation temperature",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "thinking")
    FUNCTION = "generate_image"
    CATEGORY = "BAGEL/Core"

    @classmethod
    def VALIDATE_INPUTS(
        cls, model, prompt, seed, image_ratio, cfg_text_scale, num_timesteps, **kwargs
    ):
        """Validate input parameters"""
        if not isinstance(prompt, str) or not prompt.strip():
            return "Prompt must be a non-empty string"

        if not isinstance(seed, int) or seed < 0:
            return "Seed must be a non-negative integer"

        if image_ratio not in ["1:1", "4:3", "3:4", "16:9", "9:16"]:
            return f"Invalid image_ratio: {image_ratio}"

        if (
            not isinstance(cfg_text_scale, (int, float))
            or cfg_text_scale < 1.0
            or cfg_text_scale > 8.0
        ):
            return "cfg_text_scale must be between 1.0 and 8.0"

        if (
            not isinstance(num_timesteps, int)
            or num_timesteps < 10
            or num_timesteps > 100
        ):
            return "num_timesteps must be between 10 and 100"

        return True

    def generate_image(
        self,
        model: Dict[str, Any],
        prompt: str,
        seed: int,
        image_ratio: str,
        cfg_text_scale: float,
        num_timesteps: int,
        show_thinking: bool = False,
        cfg_interval: float = 0.4,
        timestep_shift: float = 3.0,
        cfg_renorm_min: float = 1.0,
        cfg_renorm_type: str = "global",
        text_temperature: float = 0.3,
    ) -> Tuple[torch.Tensor, str]:
        """
        Generate image from text using BAGEL model

        Args:
            model: BAGEL model dictionary
            prompt: Text prompt
            seed: Random seed
            image_ratio: Image aspect ratio
            cfg_text_scale: CFG text scaling
            num_timesteps: Denoising steps
            show_thinking: Whether to display the reasoning process
            cfg_interval: CFG interval start value
            timestep_shift: Timestep offset
            cfg_renorm_min: CFG re-normalization minimum value
            cfg_renorm_type: CFG re-normalization type
            text_temperature: Text generation temperature

        Returns:
            Generated image tensor and reasoning process text
        """
        try:
            # Set random seed
            set_seed(seed)

            # Get inferencer
            inferencer = model["inferencer"]

            # Set image dimensions
            image_shapes_map = {
                "1:1": (1024, 1024),
                "4:3": (768, 1024),
                "3:4": (1024, 768),
                "16:9": (576, 1024),
                "9:16": (1024, 576),
            }
            image_shapes = image_shapes_map[image_ratio]

            # Set inference hyperparameters
            inference_hyper = {
                "max_think_token_n": 1024 if show_thinking else 1024,
                "do_sample": False if not show_thinking else False,
                "text_temperature": text_temperature if show_thinking else 0.3,
                "cfg_text_scale": cfg_text_scale,
                "cfg_interval": [cfg_interval, 1.0],  # End value fixed at 1.0
                "timestep_shift": timestep_shift,
                "num_timesteps": num_timesteps,
                "cfg_renorm_min": cfg_renorm_min,
                "cfg_renorm_type": cfg_renorm_type,
                "image_shapes": image_shapes,
            }

            # Call inferencer
            result = inferencer(text=prompt, think=show_thinking, **inference_hyper)

            # Convert image format
            pil_image = result["image"]
            tensor_image = pil_to_tensor(pil_image)

            # Get reasoning process
            thinking_text = result.get("text", "") if show_thinking else ""

            print(f"Generated image with size: {pil_image.size}")

            return (tensor_image, thinking_text)

        except Exception as e:
            print(f"Error in text to image generation: {e}")
            # Return empty image and error message
            empty_image = torch.zeros((1, 512, 512, 3))
            return (empty_image, f"Error: {str(e)}")


class BagelImageEdit:
    """BAGEL Image Edit Node"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BAGEL_MODEL", {"tooltip": "BAGEL model"}),
                "image": ("IMAGE", {"tooltip": "Input image"}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Edit the image according to the description",
                        "tooltip": "Editing prompt",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000000,
                        "tooltip": "Random seed, 0 for random",
                    },
                ),
                "cfg_text_scale": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 1.0,
                        "max": 8.0,
                        "step": 0.1,
                        "tooltip": "CFG text scaling",
                    },
                ),
                "cfg_img_scale": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 1.0,
                        "max": 4.0,
                        "step": 0.1,
                        "tooltip": "CFG image scaling",
                    },
                ),
                "num_timesteps": (
                    "INT",
                    {
                        "default": 50,
                        "min": 10,
                        "max": 100,
                        "step": 5,
                        "tooltip": "Denoising steps",
                    },
                ),
            },
            "optional": {
                "show_thinking": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Display reasoning process"},
                ),
                "cfg_interval": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": "CFG interval start value",
                    },
                ),
                "timestep_shift": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 1.0,
                        "max": 10.0,
                        "step": 0.5,
                        "tooltip": "Timestep offset",
                    },
                ),
                "cfg_renorm_min": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": "CFG re-normalization minimum value",
                    },
                ),
                "cfg_renorm_type": (
                    ["global", "local", "text_channel"],
                    {"default": "text_channel", "tooltip": "CFG re-normalization type"},
                ),
                "text_temperature": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": "Text generation temperature",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "thinking")
    FUNCTION = "edit_image"
    CATEGORY = "BAGEL/Core"

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        model,
        image,
        prompt,
        seed,
        cfg_text_scale,
        cfg_img_scale,
        num_timesteps,
        **kwargs,
    ):
        """Validate input parameters"""
        if not isinstance(prompt, str) or not prompt.strip():
            return "Prompt must be a non-empty string"

        if not isinstance(seed, int) or seed < 0:
            return "Seed must be a non-negative integer"

        if (
            not isinstance(cfg_text_scale, (int, float))
            or cfg_text_scale < 1.0
            or cfg_text_scale > 8.0
        ):
            return "cfg_text_scale must be between 1.0 and 8.0"

        if (
            not isinstance(cfg_img_scale, (int, float))
            or cfg_img_scale < 1.0
            or cfg_img_scale > 4.0
        ):
            return "cfg_img_scale must be between 1.0 and 4.0"

        if (
            not isinstance(num_timesteps, int)
            or num_timesteps < 10
            or num_timesteps > 100
        ):
            return "num_timesteps must be between 10 and 100"

        return True

    def edit_image(
        self,
        model: Dict[str, Any],
        image: torch.Tensor,
        prompt: str,
        seed: int,
        cfg_text_scale: float,
        cfg_img_scale: float,
        num_timesteps: int,
        show_thinking: bool = False,
        cfg_interval: float = 0.0,
        timestep_shift: float = 3.0,
        cfg_renorm_min: float = 1.0,
        cfg_renorm_type: str = "text_channel",
        text_temperature: float = 0.3,
    ) -> Tuple[torch.Tensor, str]:
        """
        Edit image using BAGEL model

        Args:
            model: BAGEL model dictionary
            image: Input image tensor
            prompt: Editing prompt
            seed: Random seed
            cfg_text_scale: CFG text scaling
            cfg_img_scale: CFG image scaling
            num_timesteps: Denoising steps
            show_thinking: Whether to display the reasoning process
            cfg_interval: CFG interval start value
            timestep_shift: Timestep offset
            cfg_renorm_min: CFG re-normalization minimum value
            cfg_renorm_type: CFG re-normalization type
            text_temperature: Text generation temperature

        Returns:
            Edited image tensor and reasoning process text
        """
        try:
            # Set random seed
            set_seed(seed)

            # Get inferencer
            inferencer = model["inferencer"]

            # Convert image format
            pil_image = tensor_to_pil(image)
            pil_image = pil_img2rgb(pil_image)

            # Set inference hyperparameters
            inference_hyper = {
                "max_think_token_n": 1024 if show_thinking else 1024,
                "do_sample": False if not show_thinking else False,
                "text_temperature": text_temperature if show_thinking else 0.3,
                "cfg_text_scale": cfg_text_scale,
                "cfg_img_scale": cfg_img_scale,
                "cfg_interval": [cfg_interval, 1.0],  # End value fixed at 1.0
                "timestep_shift": timestep_shift,
                "num_timesteps": num_timesteps,
                "cfg_renorm_min": cfg_renorm_min,
                "cfg_renorm_type": cfg_renorm_type,
            }

            # Call inferencer
            result = inferencer(
                image=pil_image, text=prompt, think=show_thinking, **inference_hyper
            )

            # Convert image format
            edited_pil_image = result["image"]
            tensor_image = pil_to_tensor(edited_pil_image)

            # Get reasoning process
            thinking_text = result.get("text", "") if show_thinking else ""

            print(f"Edited image with size: {edited_pil_image.size}")

            return (tensor_image, thinking_text)

        except Exception as e:
            print(f"Error in image editing: {e}")
            # Return original image and error message
            return (image, f"Error: {str(e)}")


class BagelImageUnderstanding:
    """BAGEL Image Understanding Node"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BAGEL_MODEL", {"tooltip": "BAGEL model"}),
                "image": ("IMAGE", {"tooltip": "Input image"}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "What do you see in this image?",
                        "tooltip": "Question text",
                    },
                ),
            },
            "optional": {
                "show_thinking": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Display reasoning process"},
                ),
                "do_sample": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Enable sampling"},
                ),
                "text_temperature": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Text generation temperature",
                    },
                ),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 4096,
                        "step": 64,
                        "tooltip": "Maximum new tokens",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "understand_image"
    CATEGORY = "BAGEL/Core"

    @classmethod
    def VALIDATE_INPUTS(cls, model, image, prompt, **kwargs):
        """Validate input parameters"""
        if not isinstance(prompt, str) or not prompt.strip():
            return "Prompt must be a non-empty string"

        # Validate optional parameters
        if "text_temperature" in kwargs:
            temp = kwargs["text_temperature"]
            if not isinstance(temp, (int, float)) or temp < 0.0 or temp > 1.0:
                return "text_temperature must be between 0.0 and 1.0"

        if "max_new_tokens" in kwargs:
            tokens = kwargs["max_new_tokens"]
            if not isinstance(tokens, int) or tokens < 64 or tokens > 4096:
                return "max_new_tokens must be between 64 and 4096"

        return True

    def understand_image(
        self,
        model: Dict[str, Any],
        image: torch.Tensor,
        prompt: str,
        show_thinking: bool = False,
        do_sample: bool = False,
        text_temperature: float = 0.3,
        max_new_tokens: int = 512,
    ) -> Tuple[str]:
        """
        Use BAGEL model to understand image and answer questions

        Args:
            model: BAGEL model dictionary
            image: Input image tensor
            prompt: Question text
            show_thinking: Whether to display the reasoning process
            do_sample: Whether to enable sampling
            text_temperature: Text generation temperature
            max_new_tokens: Maximum new tokens

        Returns:
            Answer text
        """
        try:
            # Get inferencer
            inferencer = model["inferencer"]

            # Convert image format
            pil_image = tensor_to_pil(image)
            pil_image = pil_img2rgb(pil_image)

            # Set inference hyperparameters
            inference_hyper = {
                "do_sample": do_sample,
                "text_temperature": text_temperature,
                "max_think_token_n": max_new_tokens,
            }

            # Call inferencer
            result = inferencer(
                image=pil_image,
                text=prompt,
                think=show_thinking,
                understanding_output=True,
                **inference_hyper,
            )

            answer_text = result["text"]

            print(f"Image understanding completed, response length: {len(answer_text)}")

            return (answer_text,)

        except Exception as e:
            print(f"Error in image understanding: {e}")
            return (f"Error: {str(e)}",)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "BagelModelLoader": BagelModelLoader,
    "BagelTextToImage": BagelTextToImage,
    "BagelImageEdit": BagelImageEdit,
    "BagelImageUnderstanding": BagelImageUnderstanding,
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "BagelModelLoader": "BAGEL Model Loader",
    "BagelTextToImage": "BAGEL Text to Image",
    "BagelImageEdit": "BAGEL Image Edit",
    "BagelImageUnderstanding": "BAGEL Image Understanding",
}

# Export for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
