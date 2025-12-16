from huggingface_hub import snapshot_download

# download  the dataset from Hugging Face Hub
snapshot_download(
    repo_id="HanzhouLiu/FloodSQL-Bench",
    local_dir="data",
    repo_type="dataset"
)
