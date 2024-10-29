from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="facebook/opt-1.3b", filename="pytorch_embs.pt", local_dir="~/")
