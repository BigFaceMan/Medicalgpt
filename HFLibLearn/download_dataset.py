from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="shibing624/medical",
    repo_type="dataset",
    local_dir="/lfs3/users/spsong/dataset/LLMData",
    max_workers=8
)