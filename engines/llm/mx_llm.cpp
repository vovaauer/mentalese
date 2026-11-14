#include "mx_llm.h"

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

struct mx_llm {
    llama_model *       model = nullptr;
    llama_context *     ctx   = nullptr;
    const llama_vocab * vocab = nullptr;
    mtmd_context *      mtmd  = nullptr;
    llama_sampler *     smpl  = nullptr;
    int                 n_batch = 512;
    std::string         out;
};

static thread_local std::string g_error;
static mx_log_cb g_log = nullptr;

static void on_log(ggml_log_level level, const char * text, void *) {
    // CONT continues the previous message (e.g. the dots of a progress line) at its level
    static thread_local ggml_log_level last = GGML_LOG_LEVEL_INFO;
    if (level == GGML_LOG_LEVEL_CONT) level = last;
    else last = level;
    if (g_log && (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) && text && *text) g_log((int) level, text);
}

static void init_once() {
    static std::once_flag once;
    std::call_once(once, [] {
        llama_log_set(on_log, nullptr);
        mtmd_helper_log_set(on_log, nullptr);
        llama_backend_init();
    });
}

static const char * fail(const std::string & msg) {
    g_error = msg;
    return nullptr;
}

extern "C" {

int mx_llm_abi(void) { return MX_LLM_ABI; }

void mx_llm_log(mx_log_cb cb) { g_log = cb; }

const char * mx_llm_marker(void) { return mtmd_default_marker(); }

const char * mx_llm_last_error(void) { return g_error.c_str(); }

const char * mx_llm_devices(void) {
    init_once();
    static thread_local std::string s;
    s.clear();
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        s += ggml_backend_dev_name(dev);
        s += '\t';
        s += ggml_backend_dev_description(dev);
        s += '\t' + std::to_string(free) + '\t' + std::to_string(total) + '\n';
    }
    return s.c_str();
}

mx_llm * mx_llm_load(const char * model, const char * mmproj, int mmproj_gpu, int n_ctx, int n_batch,
                     int n_ubatch, int n_threads) {
    init_once();
    auto * m = new mx_llm();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    m->model = llama_model_load_from_file(model, mp);
    if (!m->model) { mx_llm_free(m); fail(std::string("can't load model ") + model); return nullptr; }
    m->vocab = llama_model_get_vocab(m->model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_batch;
    cp.n_ubatch = n_ubatch;
    cp.n_seq_max = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.no_perf = true;
    if (n_threads > 0) cp.n_threads = cp.n_threads_batch = n_threads;
    m->ctx = llama_init_from_model(m->model, cp);
    if (!m->ctx) { mx_llm_free(m); fail("can't create a context (out of memory?)"); return nullptr; }
    m->n_batch = n_batch;

    if (mmproj) {
        mtmd_context_params pp = mtmd_context_params_default();
        pp.use_gpu = mmproj_gpu != 0;
        pp.print_timings = false;
        pp.warmup = false;  // warming up sizes the encoder buffers for its longest input and keeps them
        pp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        if (n_threads > 0) pp.n_threads = n_threads;
        m->mtmd = mtmd_init_from_file(mmproj, m->model, pp);
        if (!m->mtmd) { mx_llm_free(m); fail(std::string("can't load ") + mmproj); return nullptr; }
    }

    m->smpl = llama_sampler_init_greedy();
    return m;
}

const char * mx_llm_generate(mx_llm * m, const char * prompt, const float * audio, int n_audio, int max_tokens) {
    llama_memory_clear(llama_get_memory(m->ctx), true);
    llama_sampler_reset(m->smpl);
    const int n_ctx = (int) llama_n_ctx(m->ctx);
    llama_pos n_past = 0;

    if (audio) {
        if (!m->mtmd) return fail("this model has no audio projector");
        mtmd_bitmap * bmp = mtmd_bitmap_init_from_audio((size_t) n_audio, audio);
        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        mtmd_input_text text = { prompt, strlen(prompt), true, true };
        const mtmd_bitmap * bmps[] = { bmp };
        int rc = mtmd_tokenize(m->mtmd, chunks, &text, bmps, 1);
        if (rc == 0 && (int) mtmd_helper_get_n_pos(chunks) >= n_ctx) rc = -100;
        if (rc == 0) rc = mtmd_helper_eval_chunks(m->mtmd, m->ctx, chunks, 0, 0, m->n_batch, true, &n_past);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
        if (rc == -100) return fail("audio too long for the context");
        if (rc) return fail("audio prompt failed: " + std::to_string(rc));
    } else {
        const int len = (int) strlen(prompt);
        std::vector<llama_token> toks(len + 8);
        int n = llama_tokenize(m->vocab, prompt, len, toks.data(), (int) toks.size(), true, true);
        if (n < 0) return fail("tokenize failed");
        if (n >= n_ctx) return fail("prompt too long for the context");
        for (int i = 0; i < n; i += m->n_batch) {
            int k = std::min(m->n_batch, n - i);
            if (llama_decode(m->ctx, llama_batch_get_one(toks.data() + i, k))) return fail("decode failed");
        }
        n_past = n;
    }

    m->out.clear();
    char piece[256];
    for (int i = 0; i < max_tokens && n_past < n_ctx; i++, n_past++) {
        llama_token t = llama_sampler_sample(m->smpl, m->ctx, -1);
        if (llama_vocab_is_eog(m->vocab, t)) break;
        int n = llama_token_to_piece(m->vocab, t, piece, sizeof piece, 0, false);
        if (n > 0) m->out.append(piece, n);
        if (llama_decode(m->ctx, llama_batch_get_one(&t, 1))) return fail("decode failed");
    }
    return m->out.c_str();
}

void mx_llm_free(mx_llm * m) {
    if (!m) return;
    if (m->smpl) llama_sampler_free(m->smpl);
    if (m->mtmd) mtmd_free(m->mtmd);
    if (m->ctx) llama_free(m->ctx);
    if (m->model) llama_model_free(m->model);
    delete m;
}

}  // extern "C"
