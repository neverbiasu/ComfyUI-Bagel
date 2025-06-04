import os
import sys
import torch
import numpy as np
import random
import subprocess
from typing import Dict, Tuple, Optional, Any, Union
from PIL import Image
from folder_paths import folder_names_and_paths, models_dir as comfy_models_dir
from comfy.utils import ProgressBar

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import BAGEL related modules
try:
    from accelerate import (
        infer_auto_device_map,
        load_checkpoint_and_dispatch,
        init_empty_weights,
        dispatch_model,
    )
    from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
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
    """
    Unified BAGEL Model Loader with Dynamic Quantization Support

    Supports both standard BAGEL and DFloat11 models with optional quantization
    for ByteDance-Seed/BAGEL-7B-MoT model.
    """

    SUPPORTED_MODEL_REPOS = [
        "ByteDance-Seed/BAGEL-7B-MoT",  # Standard BAGEL - supports quantization
        "DFloat11/BAGEL-7B-MoT-DF11",  # DFloat11 Quantized - no additional quantization
    ]

    QUANTIZATION_MODES = {
        "BF16": "Standard (FP16/BF16)",
        "NF4": "NF4 (4-bit Quantization)",
        "INT8": "INT8 (8-bit Quantization)",
    }

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model_repo_id": (
                    cls.SUPPORTED_MODEL_REPOS,
                    {
                        "default": cls.SUPPORTED_MODEL_REPOS[0],
                        "tooltip": "Choose BAGEL model: Standard supports quantization, DFloat11 is pre-quantized",
                    },
                ),
            }
        }
        # Add quantization options for ByteDance-Seed/BAGEL-7B-MoT
        inputs["required"]["quantization_mode"] = (
            list(cls.QUANTIZATION_MODES.keys()),
            {
                "default": "BF16",
                "tooltip": "Quantization: BF16=Standard, NF4=4-bit, INT8=8-bit (Only for ByteDance model)",
            },
        )

        # Optional advanced settings
        inputs["optional"] = {
            "max_memory_gb": (
                "FLOAT",
                {
                    "default": 24.0,
                    "min": 8.0,
                    "max": 80.0,
                    "step": 1.0,
                    "tooltip": "Maximum VRAM usage per GPU in GB (for quantized models)",
                },
            ),
            "offload_folder": (
                "STRING",
                {
                    "default": "offload",
                    "tooltip": "Folder for CPU offloading (relative to ComfyUI root)",
                },
            ),
        }

        return inputs

    RETURN_TYPES = ("BAGEL_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "BAGEL/Core"

    @classmethod
    def VALIDATE_INPUTS(cls, model_repo_id, quantization_mode="BF16", **kwargs):
        """Validate input parameters"""
        if model_repo_id not in cls.SUPPORTED_MODEL_REPOS:
            return f"Unsupported model_repo_id: {model_repo_id}. Supported: {cls.SUPPORTED_MODEL_REPOS}"

        if "DFloat11" in model_repo_id and DFloat11Model is None:
            return "DFloat11 model selected, but DFloat11Model library is not installed or failed to import. Please install it: pip install dfloat11"

        # Validate quantization mode
        if quantization_mode not in cls.QUANTIZATION_MODES:
            return f"Invalid quantization_mode: {quantization_mode}. Supported: {list(cls.QUANTIZATION_MODES.keys())}"

        # DFloat11 models ignore quantization_mode (they're already quantized)
        if "DFloat11" in model_repo_id and quantization_mode != "BF16":
            print(
                f"Warning: DFloat11 model ignores quantization_mode {quantization_mode}. Using built-in quantization."
            )

        # Check for bitsandbytes availability when using quantization on ByteDance model
        if model_repo_id == "ByteDance-Seed/BAGEL-7B-MoT" and quantization_mode in [
            "NF4",
            "INT8",
        ]:
            try:
                import bitsandbytes as bnb
            except ImportError:
                return f"Quantization mode {quantization_mode} requires bitsandbytes. Please install: pip install bitsandbytes"

        return True

    def load_model(
        self,
        model_repo_id: str,
        quantization_mode: str = "BF16",
        max_memory_gb: float = 24.0,
        offload_folder: str = "offload",
    ) -> Tuple[Dict[str, Any]]:
        """
        Load BAGEL model with unified interface supporting both standard and DFloat11 models.
        Quantization is applied only to ByteDance-Seed/BAGEL-7B-MoT model.

        Args:
            model_repo_id: Hugging Face model repository ID
            quantization_mode: 1=Standard, 2=NF4, 3=INT8 (ignored for DFloat11)
            max_memory_gb: Maximum VRAM per GPU in GB
            offload_folder: Folder for CPU offloading

        Returns:
            Dictionary containing all model components
        """
        try:
            is_df11_model = model_repo_id == "DFloat11/BAGEL-7B-MoT-DF11"
            is_standard_model = model_repo_id == "ByteDance-Seed/BAGEL-7B-MoT"
            # For DFloat11, always use standard mode (it's already quantized)
            if is_df11_model and quantization_mode != "BF16":
                print(
                    f"DFloat11 model detected. Ignoring quantization_mode {quantization_mode} and using built-in quantization."
                )
                quantization_mode = "BF16"

            print(
                f"Loading {model_repo_id} with {self.QUANTIZATION_MODES[quantization_mode]} mode..."
            )

            base_repo_dir = os.path.join(comfy_models_dir, "bagel")
            repo_name_segment = model_repo_id.split("/")[-1]
            local_model_dir = os.path.join(base_repo_dir, repo_name_segment)

            common_vae_dir = os.path.join(comfy_models_dir, "vae")
            common_vae_file = os.path.join(common_vae_dir, "ae.safetensors")
            # Check if model exists locally, if not, download it
            if not os.path.exists(local_model_dir) or not check_model_files(
                local_model_dir, is_df11_model
            ):
                print(
                    f"Core model files not found or incomplete for {model_repo_id} at {local_model_dir}. Attempting download..."
                )

                # Attempt to download using huggingface_hub
                downloaded_path = download_model_with_hf_hub(
                    local_model_dir, repo_id=model_repo_id
                )
                if not downloaded_path:
                    raise FileNotFoundError(
                        f"Failed to download BAGEL model. Please manually download it from "
                        f"{model_repo_id} and place it in {local_model_dir}"
                    )

                print(f"Successfully downloaded BAGEL model to {local_model_dir}")

            # Final check that all required files exist
            if not check_model_files(local_model_dir, is_df11_model):
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

            if is_df11_model:
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
            elif is_standard_model:
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

                # Initialize empty model
                with init_empty_weights():
                    language_model = Qwen2ForCausalLM(llm_config)
                    vit_model = SiglipVisionModel(vit_config)
                    model = Bagel(language_model, vit_model, config)
                    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
                        vit_config, meta=True
                    )

                # Load tokenizer
                tokenizer = Qwen2Tokenizer.from_pretrained(local_model_dir)
                tokenizer, new_token_ids, _ = add_special_tokens(
                    tokenizer
                )  # Create transforms
                vae_transform = ImageTransform(1024, 512, 16)
                vit_transform = ImageTransform(980, 224, 14)

                # Setup device mapping for quantized models
                if quantization_mode in ["NF4", "INT8"]:
                    max_memory_per_gpu = f"{max_memory_gb}GiB"
                    device_map = infer_auto_device_map(
                        model,
                        max_memory={
                            i: max_memory_per_gpu
                            for i in range(torch.cuda.device_count())
                        },
                        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
                    )

                    # Ensure same device for critical modules
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

                # Load model based on quantization mode
                checkpoint_path = os.path.join(local_model_dir, "ema.safetensors")

                if quantization_mode == "BF16":
                    # Standard loading
                    print("Loading model in standard FP16/BF16 mode...")
                    model = load_checkpoint_and_dispatch(
                        model,
                        checkpoint=checkpoint_path,
                        device_map="auto",
                        dtype=torch.bfloat16,
                        force_hooks=True,
                    ).eval()

                elif quantization_mode == "NF4":
                    # NF4 quantization
                    print("Loading model with NF4 (4-bit) quantization...")
                    bnb_quantization_config = BnbQuantizationConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=False,
                        bnb_4bit_quant_type="nf4",
                    )
                    model = load_and_quantize_model(
                        model,
                        weights_location=checkpoint_path,
                        bnb_quantization_config=bnb_quantization_config,
                        device_map=device_map,
                        offload_folder=offload_folder,
                    ).eval()

                elif quantization_mode == "INT8":
                    # INT8 quantization with improved memory management
                    print("Loading model with INT8 (8-bit) quantization...")

                    # Clear GPU cache before quantization
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Use more conservative device mapping for INT8
                    reduced_memory_gb = max(
                        8.0, max_memory_gb * 0.7
                    )  # Use 70% of max memory
                    reduced_device_map = infer_auto_device_map(
                        model,
                        max_memory={
                            i: f"{reduced_memory_gb}GiB"
                            for i in range(torch.cuda.device_count())
                        },
                        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
                    )

                    # Apply same device constraints
                    if torch.cuda.device_count() == 1:
                        first_device = reduced_device_map.get(
                            same_device_modules[0], "cuda:0"
                        )
                        for k in same_device_modules:
                            if k in reduced_device_map:
                                reduced_device_map[k] = first_device
                            else:
                                reduced_device_map[k] = "cuda:0"
                    else:
                        first_device = reduced_device_map.get(same_device_modules[0])
                        for k in same_device_modules:
                            if k in reduced_device_map:
                                reduced_device_map[k] = first_device

                    bnb_quantization_config = BnbQuantizationConfig(
                        load_in_8bit=True,
                        torch_dtype=torch.bfloat16,
                        llm_int8_threshold=6.0,  # Higher threshold for better stability
                        llm_int8_skip_modules=[
                            "vit_model",
                            "vae2llm",
                            "llm2vae",
                        ],  # Skip vision modules
                    )

                    model = load_and_quantize_model(
                        model,
                        weights_location=checkpoint_path,
                        bnb_quantization_config=bnb_quantization_config,
                        device_map=reduced_device_map,
                        offload_folder=offload_folder,
                    ).eval()

                else:
                    raise ValueError(
                        f"Unsupported quantization mode: {quantization_mode}"
                    )

                # Create inferencer
                inferencer = InterleaveInferencer(
                    model=model,
                    vae_model=vae_model,
                    tokenizer=tokenizer,
                    vae_transform=vae_transform,
                    vit_transform=vit_transform,
                    new_token_ids=new_token_ids,
                )

                # Prepare model dictionary with quantization info
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
                    "quantization_mode": quantization_mode,
                    "quantization_info": self.QUANTIZATION_MODES[quantization_mode],
                    "is_df11": False,
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                }

                print(
                    f"Successfully loaded BAGEL model with {self.QUANTIZATION_MODES[quantization_mode]} mode"
                )
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
        cfg_renorm_min: float = 0.0,
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

            # Initialize ProgressBar
            # The loop in Bagel.generate_image runs for (num_timesteps - 1) iterations
            actual_iterations = num_timesteps - 1 if num_timesteps > 0 else 0
            pbar = ProgressBar(actual_iterations)

            # Call inferencer, passing pbar
            result = inferencer(
                text=prompt, think=show_thinking, pbar=pbar, **inference_hyper
            )

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
                        "default": 0.0,
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
        cfg_renorm_min: float = 0.0,
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

            # Initialize ProgressBar
            actual_iterations = num_timesteps - 1 if num_timesteps > 0 else 0
            pbar = ProgressBar(actual_iterations)

            # Call inferencer, passing pbar
            result = inferencer(
                image=pil_image,
                text=prompt,
                think=show_thinking,
                pbar=pbar,
                **inference_hyper,
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
