# ohseeare

A cross-platform screen-region OCR tool powered by [GLM-Edge](https://huggingface.co/THUDM/glm-edge-v-5b) via [llama.cpp](https://github.com/ggml-org/llama.cpp).

Capture any region of your screen with a snipping-tool-style overlay, and the selected image is fed through a vision language model to extract text.

## Features

- **Built-in screen capture** — cross-platform ImGui overlay (no external tools like Flameshot required)
- **Global hotkey** — `Ctrl+Shift+S` triggers capture (Linux/X11)
- **CLI + GUI** — use `ohseeare` for scripting or `ohseeare-gui` for interactive use
- **Copy to clipboard** — one-click copy of OCR results

## Dependencies

### Build requirements

| Dependency | Notes |
|------------|-------|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Source tree — passed to CMake via `-DLLAMA_CPP_DIR` |
| [GLFW](https://www.glfw.org/) ≥ 3.3 | Windowing / input |
| [libepoxy](https://github.com/anholt/libepoxy) | OpenGL dispatch |
| [libpng](http://www.libpng.org/) | Screenshot PNG encoding |
| [ImGui](https://github.com/ocornut/imgui) | Immediate-mode GUI (downloaded automatically via CMake FetchContent) |
| CMake ≥ 3.14 | Build system |
| C++17 compiler | GCC, Clang, or MSVC |

### Platform-specific

| Platform | Additional |
|----------|-----------|
| Linux | X11 development headers (`libx11-dev`) |
| macOS | CoreGraphics framework (included with Xcode) |
| Windows | GDI (included with Windows SDK) |

### Models

Download the GGUF models from Hugging Face:

```sh
mkdir -p models/glm-ocr
huggingface-cli download THUDM/glm-edge-v-5b-gguf \
  glm-ocr.gguf mmproj-glm-ocr.gguf \
  --local-dir models/glm-ocr
```

This places two files under `models/glm-ocr/`:
- `glm-ocr.gguf` — the main vision-language model (~900 MB)
- `mmproj-glm-ocr.gguf` — the multimodal projector (~830 MB)

> If you prefer a smaller quantised variant (e.g. `GLM-OCR.i1-Q6_K.gguf`),
> download that file instead and pass its path via `-DOHSEEARE_MODEL_PATH=...`.

## Building

### 1. Clone this repository

```sh
git clone https://github.com/YOUR_USERNAME/ohseeare.git
cd ohseeare
```

### 2. Initialise submodules

```sh
git submodule update --init --recursive
```

### 3. Configure and build

```sh
cmake -B build -DLLAMA_CPP_DIR=/path/to/llama.cpp
cmake --build build -j$(nproc)
```

#### Optional CMake variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_CPP_DIR` | *(required)* | Path to llama.cpp source tree |
| `OHSEEARE_MODEL_PATH` | `models/glm-ocr/glm-ocr.gguf` | Path to main GGUF model |
| `OHSEEARE_MMPROJ_PATH` | `models/glm-ocr/mmproj-glm-ocr.gguf` | Path to vision projector GGUF |
| `GGML_CUDA` | `OFF` | Enable CUDA/GPU acceleration (requires CUDA toolkit) |

Example with CUDA and custom model paths:

```sh
cmake -B build \
  -DLLAMA_CPP_DIR=/path/to/llama.cpp \
  -DGGML_CUDA=ON \
  -DOHSEEARE_MODEL_PATH=/path/to/model.gguf \
  -DOHSEEARE_MMPROJ_PATH=/path/to/mmproj.gguf
```

## Usage

### GUI

```sh
./build/bin/ohseeare-gui
```

1. Click **Capture Region** (or press `Ctrl+Shift+S` on Linux)
2. Click and drag to select a screen region
3. Wait for OCR to complete
4. Click **Copy** to copy the result to the clipboard

### CLI

```sh
./build/bin/ohseeare \
  -m models/glm-ocr/glm-ocr.gguf \
  --mmproj models/glm-ocr/mmproj-glm-ocr.gguf \
  --image screenshot.png \
  -p "Convert the image to text."
```

## Project structure

```
ohseeare/
├── CMakeLists.txt          # Top-level build (references llama.cpp)
├── README.md
├── LICENSE
├── fonts/                  # Bundled fonts (Inter, JetBrainsMono)
├── vendor/
│   └── screen-capture-overlay/  # Git submodule — cross-platform capture lib
├── src/
│   ├── ohseeare.cpp        # CLI binary
│   ├── ohseeare-gui.cpp    # GUI binary
│   ├── tinyfiledialogs.h   # Vendored file dialog library (Zlib)
│   └── tinyfiledialogs.c
└── _deps/                  # (CMake FetchContent — not committed)
```

## License

MIT — see [LICENSE](LICENSE).

Vendored dependencies:
- [tinyfiledialogs](https://sourceforge.net/projects/tinyfiledialogs/) — Zlib license
- [ImGui](https://github.com/ocornut/imgui) — MIT license (downloaded at build time)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — MIT license
