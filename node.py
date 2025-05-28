import os
import sys
import torch
import numpy as np
import random
import subprocess
from typing import Dict, Tuple, Optional, Any, Union
from PIL import Image
from folder_paths import folder_names_and_paths

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import BAGEL related modules
try:
    from accelerate import (
        infer_auto_device_map,
        load_checkpoint_and_dispatch,
        init_empty_weights,
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

# Register the BAGEL model folder
models_dir = os.path.join(os.getcwd(), "models")
folder_names_and_paths["bagel"] = (
    [os.path.join(models_dir, "bagel")],
    [".json", ".safetensors"],
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


def check_model_files(model_path: str) -> bool:
    """
    Check if all required model files exist

    Args:
        model_path: Path to the model directory

    Returns:
        True if all files exist, False otherwise
    """
    required_files = [
        "llm_config.json",
        "vit_config.json",
        "ae.safetensors",
        "ema.safetensors",
    ]

    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
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

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (
                    "STRING",
                    {
                        "default": "ByteDance-Seed/BAGEL-7B-MoT",
                        "tooltip": "Hugging Face model repo name or local path",
                    },
                ),
            }
        }

    RETURN_TYPES = ("BAGEL_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "BAGEL/Core"

    @classmethod
    def VALIDATE_INPUTS(cls, model_path):
        """Validate input parameters"""
        if not isinstance(model_path, str) or not model_path.strip():
            return "Model path must be a non-empty string"

        return True

    def load_model(self, model_path: str) -> Tuple[Dict[str, Any]]:
        """
        Load BAGEL model and its components. Automatically download the model if not found.

        Args:
            model_path: URL to the Hugging Face model repository

        Returns:
            Dictionary containing all model components
        """
        try:
            # Define base model directory
            base_model_dir = os.path.join(os.getcwd(), "models", "bagel")

            # Extract repo name from model_path for the subdirectory
            repo_name = model_path.split("/")[-1] if "/" in model_path else model_path
            local_model_dir = os.path.join(base_model_dir, repo_name)

            # Check if model exists locally, if not, download it
            if not os.path.exists(local_model_dir) or not check_model_files(
                local_model_dir
            ):
                print(
                    f"Model not found locally. Attempting to download from {model_path}..."
                )

                # Attempt to download using huggingface_hub
                downloaded_path = download_model_with_hf_hub(
                    local_model_dir, repo_id=model_path
                )
                if not downloaded_path:
                    raise FileNotFoundError(
                        f"Failed to download BAGEL model. Please manually download it from "
                        f"{model_path} and place it in {local_model_dir}"
                    )

                print(f"Successfully downloaded BAGEL model to {local_model_dir}")

            # Final check that all required files exist
            if not check_model_files(local_model_dir):
                raise FileNotFoundError(
                    f"Required model files missing in {local_model_dir}"
                )

            # Load configuration files
            llm_config = Qwen2Config.from_json_file(
                os.path.join(local_model_dir, "llm_config.json")
            )
            llm_config.qk_norm = True
            llm_config.tie_word_embeddings = False
            llm_config.layer_module = "Qwen2MoTDecoderLayer"

            vit_config = SiglipVisionConfig.from_json_file(
                os.path.join(local_model_dir, "vit_config.json")
            )
            vit_config.rope = False
            vit_config.num_hidden_layers -= 1

            vae_model, vae_config = load_ae(
                local_path=os.path.join(local_model_dir, "ae.safetensors")
            )

            # Create BAGEL configuration
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

            # Initialize model
            with init_empty_weights():
                language_model = Qwen2ForCausalLM(llm_config)
                vit_model = SiglipVisionModel(vit_config)
                model = Bagel(language_model, vit_model, config)
                model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
                    vit_config, meta=True
                )

            # Load tokenizer
            tokenizer = Qwen2Tokenizer.from_pretrained(local_model_dir)
            tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

            # Create transformers
            vae_transform = ImageTransform(1024, 512, 16)
            vit_transform = ImageTransform(980, 224, 14)
            model_device_final_str = "cpu"

            if is_df11_model:
                vae_model, vae_config = None, None
                potential_vae_paths = [
                    os.path.join(local_model_dir, "vae", "ae.safetensors"),
                    os.path.join(local_model_dir, "ae.safetensors"),
                    common_vae_file,
                ]
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
                if not vae_loaded_path:
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
                model = model.to(torch.bfloat16)
                model.load_state_dict(
                    {
                        name: (
                            torch.empty(param.shape, dtype=param.dtype, device="cpu")
                            if param.device.type == "meta"
                            else param
                        )
                        for name, param in model.state_dict().items()
                    },
                    assign=True,
                )
                model = DFloat11Model.from_pretrained(
                    local_model_dir,
                    bfloat16_model=model,
                    device="cpu",
                )
                device_map = infer_auto_device_map(
                    model,
                    max_memory={0: "40GiB"},
                    no_split_module_classes=[
                        "Bagel",
                        "Qwen2MoTDecoderLayer",
                        "SiglipVisionModel",
                    ],
                )
                same_device_modules = [
                    "language_model.model.embed_tokens",
                    "time_embedder",
                    "latent_pos_embed",
                    "vae2llm",
                    "llm2vae",
                    "connector",
                    "vit_pos_embed",
                ]
                if torch.cuda.device_count() == 1:
                    first_device = device_map.get(same_device_modules[0], "cuda:0")
                    for k in same_device_modules:
                        if k in device_map:
                            device_map[k] = first_device
                        else:
                            device_map[k] = "cuda:0"
                else:
                    first_device = device_map.get(same_device_modules[0])
                    for k in same_device_modules:
                        if k in device_map:
                            device_map[k] = first_device
                model = dispatch_model(model, device_map=device_map, force_hooks=True)
                model = model.eval()
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
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                }
                print(f"Successfully loaded BAGEL DF11 model from {local_model_dir}")
                return (model_dict,)

        except Exception as e:
            print(f"Error loading BAGEL model: {e}")
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
                        "default": 0.0,
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
