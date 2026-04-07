#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "chat.h"
#include "log.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <clocale>
#include <cstdio>
#include <signal.h>

static volatile bool g_is_generating = false;

static void sigint_handler(int) {
    g_is_generating = false;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    ggml_time_init();
    common_init();

    common_params params;
    params.n_predict = 2048;
    params.n_ctx     = 8192;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD, nullptr)) {
        return 1;
    }

    if (params.mmproj.path.empty() || params.image.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf --mmproj mmproj.gguf --image img.png [-p prompt]\n", argv[0]);
        return 1;
    }

    // silence all library output — only generated text goes to stdout
    params.verbosity = -1;
    common_log_set_verbosity_thold(-1);
    llama_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);
    mtmd_helper_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);

#ifdef _WIN32
    signal(SIGINT, sigint_handler);
#else
    struct sigaction sa{};
    sa.sa_handler = sigint_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, nullptr);
#endif

    // load text model
    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * lctx  = llama_init->context();
    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (!model || !lctx) { return 1; }

    // load vision projector
    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu       = params.mmproj_use_gpu;
    mparams.print_timings = false;
    mparams.n_threads     = params.cpuparams.n_threads;
    mtmd::context_ptr ctx_vision(mtmd_init_from_file(params.mmproj.path.c_str(), model, mparams));
    if (!ctx_vision) {
        fprintf(stderr, "failed to load mmproj: %s\n", params.mmproj.path.c_str());
        return 1;
    }

    // load image(s)
    mtmd::bitmaps bitmaps;
    for (const auto & path : params.image) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx_vision.get(), path.c_str()));
        if (!bmp.ptr) {
            fprintf(stderr, "failed to load image: %s\n", path.c_str());
            return 1;
        }
        bitmaps.entries.push_back(std::move(bmp));
    }

    // build user prompt with image marker(s) prepended
    std::string prompt = params.prompt.empty() ? "Convert the image to text." : params.prompt;
    for (size_t i = 0; i < params.image.size(); i++) {
        prompt = std::string(mtmd_default_marker()) + prompt;
    }

    // format via the model's chat template
    auto tmpls = common_chat_templates_init(model, params.chat_template);
    std::vector<common_chat_msg> history;
    common_chat_msg user_msg{"user", prompt, {}};
    auto formatted = common_chat_format_single(tmpls.get(), history, user_msg, true, params.use_jinja);

    // tokenize text + image
    mtmd_input_text inp{formatted.c_str(), /*add_special=*/true, /*parse_special=*/true};
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_ptr = bitmaps.c_ptr();
    if (mtmd_tokenize(ctx_vision.get(), chunks.ptr.get(), &inp,
                      bitmaps_ptr.data(), bitmaps_ptr.size()) != 0) {
        fprintf(stderr, "tokenization failed\n");
        return 1;
    }

    // eval prompt
    llama_pos n_past = 0;
    llama_batch batch = llama_batch_init(1, 0, 1);
    if (mtmd_helper_eval_chunks(ctx_vision.get(), lctx, chunks.ptr.get(),
                                n_past, 0, params.n_batch, true, &n_past)) {
        fprintf(stderr, "eval failed\n");
        llama_batch_free(batch);
        return 1;
    }

    // generate
    auto * smpl = common_sampler_init(model, params.sampling);
    int n_predict = params.n_predict < 0 ? 2048 : params.n_predict;

    g_is_generating = true;
    for (int i = 0; i < n_predict && g_is_generating; i++) {
        llama_token token = common_sampler_sample(smpl, lctx, -1);
        common_sampler_accept(smpl, token, true);

        if (llama_vocab_is_eog(vocab, token)) break;

        printf("%s", common_token_to_piece(lctx, token).c_str());
        fflush(stdout);

        common_batch_clear(batch);
        common_batch_add(batch, token, n_past++, {0}, true);
        if (llama_decode(lctx, batch)) {
            fprintf(stderr, "decode failed\n");
            break;
        }
    }
    printf("\n");

    common_sampler_free(smpl);
    llama_batch_free(batch);
    return 0;
}
