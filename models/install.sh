# Download GLM-OCR is gguf format from hugging face
# https://huggingface.co/ggml-org/GLM-OCR-GGUF/resolve/main/GLM-OCR-f16.gguf
#!/bin/bash

if [ -f "GLM-OCR-f16.gguf" ]; then
    echo "File already exists. Skipping download."
else
    # Download the file using wget
    echo "Downloading GLM-OCR-f16.gguf..."
    wget -O GLM-OCR-f16.gguf https://huggingface.co/ggml-org/GLM-OCR-GGUF/resolve/main/GLM-OCR-f16.gguf
    echo "Download completed."
fi
