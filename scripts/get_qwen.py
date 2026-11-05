"""Download the pinned Qwen3.5-0.8B weights used for patch conformance. Prints the snapshot folder."""
from huggingface_hub import snapshot_download

print(snapshot_download("Qwen/Qwen3.5-0.8B", revision="2fc06364715b967f1860aea9cf38778875588b17",
                        allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json", "tokenizer*",
                                        "generation_config.json", "chat_template*", "vocab.json", "merges.txt"]))
