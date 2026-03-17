import os
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def download_model_to_local(repo_id, local_dir):
    """
    Func to download a Hugging Face model repository to a local directory.
    """
    print(f"\nInstalling {repo_id} to local path: {local_dir}")

    try:
        # snapshot_download to download the model repository to local_dir
        model_path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.py", "*.bin"] # Skip unnecessary files to reduce size
        )
        print("Download completed")
        return True
    except Exception as e:
        print(f"Error during download: {e}")
        return False

def verify_and_load_model(local_dir):
    """
    Func to verify integrity by attempting to load the model into RAM/VRAM.
    """
    print(f"\nLoading model from: {local_dir}")
    
    # Check basic config file
    config_path = os.path.join(local_dir, "config.json")
    if not os.path.exists(config_path):
        print("Missing config.json file")
        return False
    
    # load quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Try loading Model and Tokenizer
    try:
        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(local_dir)
        print("Loading tokenizer successful")

        # Load Model (Try loading into GPU if available)
        model = AutoModelForCausalLM.from_pretrained(
            local_dir,
            device_map="auto",       # allow automatic device mapping
            trust_remote_code=False, # use local code only
            quantization_config=bnb_config,
            dtype=torch.float16,
        )
        print(f"Model Architecture: {model.config.architectures}")
        print("Verification successful! Model loaded correctly.")

        print(f"Allocated VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Reserved VRAM:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
        # Free memory after testing (to avoid unnecessary VRAM usage)
        del model
        torch.cuda.empty_cache()
        
        return True

    except Exception as e:
        print(f"Error loading model: {e}")
        return False