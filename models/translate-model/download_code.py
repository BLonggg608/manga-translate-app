import os
import gc
import torch
# if you having multiple gpus and want to set specific gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from download_hg_model_func import download_model_to_local, verify_and_load_model

HOME = os.getcwd()

# model name
REPO_ID = "IlyaGusev/gemma-2-2b-it-abliterated"

# local directory to save model
LOCAL_DIR = f"{HOME}/translate-model/models/gemma-2-2b-it-abliterated"

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def main():
    # create local dir if not exist
    if not os.path.exists(LOCAL_DIR):
        os.makedirs(LOCAL_DIR)
        success_download = download_model_to_local(REPO_ID, LOCAL_DIR)
    else:
        print(f"Model directory {LOCAL_DIR} already exists. Skipping download.")
        success_download = True

    free_memory()

    if success_download:
        verify_and_load_model(LOCAL_DIR)
    else:
        print("Download failed")

if __name__ == "__main__":
    main()