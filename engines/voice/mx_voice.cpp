#include "mx_voice.h"

#include "qwen.h"

#include <string>

static_assert(QT_ABI_VERSION == 6, "qwentts.cpp's ABI changed: review mx_voice.cpp against src/qwen.h");

struct mx_voice {
    qt_context * ctx = nullptr;
    qt_voice_ref ref = {};
};

static thread_local std::string g_error;
static mx_log_cb g_log = nullptr;

static void on_log(qt_log_level level, const char * msg, void *) {
    if (g_log && level >= QT_LOG_WARN) g_log(level == QT_LOG_WARN ? 3 : 4, msg);
}

static int fail(const char * what) {
    g_error = std::string(what) + ": " + qt_last_error();
    return -1;
}

struct chunk_ctx {
    mx_voice_chunk_cb cb;
    void * user;
};

static bool on_chunk(const float * samples, int n, void * user) {
    auto * c = static_cast<chunk_ctx *>(user);
    return c->cb(samples, n, c->user) != 0;
}

extern "C" {

int mx_voice_abi(void) { return MX_VOICE_ABI; }

void mx_voice_log(mx_log_cb cb) {
    g_log = cb;
    qt_log_set(on_log, nullptr);  // never NULL: that restores qwentts's own stderr logging
}

const char * mx_voice_last_error(void) { return g_error.c_str(); }

mx_voice * mx_voice_load(const char * talker, const char * codec, const char * device) {
    qt_log_set(on_log, nullptr);
    qt_init_params p;
    qt_init_default_params(&p);
    p.talker_path = talker;
    p.codec_path = codec;
    p.device = device;
    auto * v = new mx_voice();
    v->ctx = qt_init(&p);
    if (!v->ctx) {
        fail("can't load the voice model");
        delete v;
        return nullptr;
    }
    return v;
}

int mx_voice_set_ref(mx_voice * v, const float * wav, int n) {
    qt_voice_ref_free(&v->ref);
    if (qt_extract_voice_ref(v->ctx, wav, n, &v->ref)) return fail("can't read the voice sample");
    return 0;
}

int mx_voice_say(mx_voice * v, const char * text, const char * lang, int max_frames, const char * ref_text,
                 mx_voice_chunk_cb cb, void * user) {
    if (!v->ref.ref_spk_emb) {
        g_error = "no voice sample set";
        return -1;
    }
    qt_tts_params p;
    qt_tts_default_params(&p);
    p.text = text;
    p.lang = lang;
    p.max_new_tokens = max_frames;
    p.ref_spk_emb = v->ref.ref_spk_emb;
    p.ref_spk_dim = v->ref.ref_spk_dim;
    if (ref_text) {
        p.ref_codes = v->ref.ref_codes;
        p.ref_T = v->ref.ref_T;
        p.ref_text = ref_text;
    }
    chunk_ctx c = { cb, user };
    p.on_chunk = on_chunk;
    p.on_chunk_user_data = &c;
    qt_audio out = {};
    qt_status rc = qt_synthesize(v->ctx, &p, &out);
    qt_audio_free(&out);
    if (rc == QT_STATUS_CANCELLED) return 1;
    if (rc) return fail("speech failed");
    return 0;
}

void mx_voice_free(mx_voice * v) {
    if (!v) return;
    qt_voice_ref_free(&v->ref);
    qt_free(v->ctx);
    delete v;
}

}  // extern "C"
