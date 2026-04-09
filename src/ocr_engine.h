#pragma once
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "sampling.h"

struct OcrEngine {
  std::unique_ptr<common_init_result> llama_init;

  llama_model *model = nullptr;
  llama_context *lctx = nullptr;

  const llama_vocab *vocab = nullptr;

  mtmd::context_ptr ctx_vision{nullptr};

  std::unique_ptr<common_chat_templates, decltype(&common_chat_templates_free)>
      tmpls{nullptr, common_chat_templates_free};

  bool ok = false;

  // Load model + vision projector.
  bool load(const char *model_path, const char *mmproj_path,
            int n_threads = 8) {

    // Silence library log spam
    common_log_set_verbosity_thold(-1);
    llama_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);
    mtmd_helper_log_set([](enum ggml_log_level, const char *, void *) {},
                        nullptr);

    common_params params;
    params.model.path = model_path;
    params.mmproj.path = mmproj_path;
    params.n_predict = 1024;
    params.n_ctx = 2048;
    params.n_batch = 1024;
    params.n_ubatch = 512;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.cpuparams.n_threads = n_threads;
    params.verbosity = -1;
    params.sampling.temp = 0.1f;

    llama_init = common_init_from_params(params);
    if (!llama_init) {
      return false;
    }

    model = llama_init->model();
    lctx = llama_init->context();
    vocab = llama_model_get_vocab(model);

    if (!model || !lctx || !vocab) {
      return false;
    }

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = true;
    mparams.print_timings = false;
    mparams.n_threads = params.cpuparams.n_threads;
    ctx_vision.reset(mtmd_init_from_file(mmproj_path, model, mparams));

    if (!ctx_vision) {
      return false;
    }

    tmpls.reset(common_chat_templates_init(model, "").release());

    ok = true;
    return true;
  }

  // Free all resources so a new model can be loaded.
  void free() {
    ok = false;
    ctx_vision.reset();
    tmpls.reset(nullptr);
    llama_init.reset(); // frees model + context
    model = nullptr;
    lctx = nullptr;
    vocab = nullptr;
  }

  // Run inference on raw RGBA pixels (e.g. from sco_capture_region).
  // Strips the alpha channel before passing to the model (which expects RGB).
  std::string infer(const uint8_t *rgba, int width, int height,
                    const char *prompt_text, float temperature = 0.1f,
                    int max_tokens = 2048) {

    if (!ok) {
      return "(model not loaded)";
    }

    std::vector<uint8_t> rgb((size_t)width * height * 3);

    for (int i = 0; i < width * height; i++) {
      rgb[i * 3] = rgba[i * 4];
      rgb[i * 3 + 1] = rgba[i * 4 + 1];
      rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }

    mtmd::bitmaps bitmaps;
    {
      mtmd::bitmap bmp(mtmd_bitmap_init(width, height, rgb.data()));

      if (!bmp.ptr) {
        return "(failed to create bitmap)";
      }

      bitmaps.entries.push_back(std::move(bmp));
    }
    return infer_impl(bitmaps, prompt_text, temperature, max_tokens);
  }

private:
  std::string infer_impl(mtmd::bitmaps &bitmaps, const char *prompt_text,
                         float temperature, int max_tokens) {
    // Build the prompt with image marker
    std::string prompt = std::string(mtmd_default_marker()) + prompt_text;

    // Format via the model's chat template
    std::vector<common_chat_msg> history;
    common_chat_msg user_msg{"user", prompt, {}};
    auto formatted =
        common_chat_format_single(tmpls.get(), history, user_msg, true, false);

    // Tokenize text + image
    mtmd_input_text inp{formatted.c_str(), true, true};
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_ptr = bitmaps.c_ptr();

    if (mtmd_tokenize(ctx_vision.get(), chunks.ptr.get(), &inp,
                      bitmaps_ptr.data(), bitmaps_ptr.size()) != 0) {

      return "(tokenization failed)";
    }

    // Clear KV cache so each query starts fresh
    llama_memory_clear(llama_get_memory(lctx), true);

    // Eval prompt
    llama_pos n_past = 0;
    llama_batch batch = llama_batch_init(1, 0, 1);
    if (mtmd_helper_eval_chunks(ctx_vision.get(), lctx, chunks.ptr.get(),
                                n_past, 0, 1024, true, &n_past)) {
      llama_batch_free(batch);
      return "(eval failed)";
    }

    // Sample / generate
    common_params_sampling sparams;
    sparams.temp = temperature;
    auto *smpl = common_sampler_init(model, sparams);

    std::string out;
    for (int i = 0; i < max_tokens; i++) {
      llama_token token = common_sampler_sample(smpl, lctx, -1);
      common_sampler_accept(smpl, token, true);

      if (llama_vocab_is_eog(vocab, token)) {
        break;
      }

      out += common_token_to_piece(lctx, token);

      common_batch_clear(batch);
      common_batch_add(batch, token, n_past++, {0}, true);
      if (llama_decode(lctx, batch)) {
        break;
      }
    }

    common_sampler_free(smpl);
    llama_batch_free(batch);

    // Strip trailing whitespace
    while (!out.empty() &&
           (out.back() == '\n' || out.back() == '\r' || out.back() == ' ')) {

      out.pop_back();
    }

    return out;
  }
};
