# ohseeare

Screen-region OCR powered by [GLM-Edge](https://huggingface.co/THUDM/glm-edge-v-5b) via [llama.cpp](https://github.com/ggml-org/llama.cpp). Draw a box around any part of your screen and get the text back.

## Dependencies

| Dependency | Notes |
|------------|-------|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Source tree — passed via `-DLLAMA_CPP_DIR` |
| [GLFW](https://www.glfw.org/) ≥ 3.3 | `libglfw3-dev` on Debian/Ubuntu |
| X11 headers | `libx11-dev` on Debian/Ubuntu (Linux only) |
| CMake ≥ 3.14, C++17 | |

ImGui and GLAD are bundled in-tree.

## Models

```sh
mkdir -p models/glm-ocr
huggingface-cli download THUDM/glm-edge-v-5b-gguf \
  glm-ocr.gguf mmproj-glm-ocr.gguf \
  --local-dir models/glm-ocr
```

## Building

```sh
cmake -B build -DLLAMA_CPP_DIR=/path/to/llama.cpp
cmake --build build -j$(nproc)
```

With CUDA:

```sh
cmake -B build -DLLAMA_CPP_DIR=/path/to/llama.cpp -DGGML_CUDA=ON
cmake --build build -j$(nproc)
```

Custom model paths can be set with `-DOHSEEARE_MODEL_PATH=...` and `-DOHSEEARE_MMPROJ_PATH=...`.

## Usage

### GUI

```sh
./build/ohseeare-gui
```

Press `Ctrl+Shift+S` (or click **Capture Region**), drag to select, then copy the result. Use **Rerun** to re-OCR the same capture with different settings.

### CLI

```sh
./build/ohseeare \
  -m models/glm-ocr/glm-ocr.gguf \
  --mmproj models/glm-ocr/mmproj-glm-ocr.gguf \
  --image screenshot.png \
  -p "Convert the image to text."
```

## License

MIT — see [LICENSE](LICENSE).

Bundled: [tinyfiledialogs](https://sourceforge.net/projects/tinyfiledialogs/) (Zlib), [ImGui](https://github.com/ocornut/imgui) (MIT), [GLAD](https://glad.dav1d.de/) (MIT).
