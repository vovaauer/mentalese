// mentalese's C ABI over qwentts.cpp (Qwen3-TTS): clone a voice from a reference recording and
// stream speech. qwentts.cpp and its ggml fork are linked in statically and kept private, so
// this can share a process with llama.cpp's ggml.
#pragma once
#include <stdint.h>

#if defined(_WIN32)
#  define MX_API __declspec(dllexport)
#else
#  define MX_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MX_VOICE_ABI 1

typedef struct mx_voice mx_voice;
typedef void (*mx_log_cb)(int level, const char * msg);  // level: 2 info, 3 warn, 4 error
// 24 kHz mono float chunk; called from the engine's worker thread. Return 0 to stop.
typedef int (*mx_voice_chunk_cb)(const float * samples, int n, void * user);

MX_API int          mx_voice_abi(void);
MX_API void         mx_voice_log(mx_log_cb cb);  // NULL: silent
// device: ggml device name (CUDA0, Vulkan0, MTL0, CPU) or NULL for the best one.
MX_API mx_voice *   mx_voice_load(const char * talker, const char * codec, const char * device);
// Reference recording (24 kHz mono float) to clone. 0 on success.
MX_API int          mx_voice_set_ref(mx_voice * v, const float * wav, int n);
// Speaks `text` in `lang` (Qwen3-TTS language name, lowercase), streaming chunks to cb; blocks
// until done. ref_text = the reference recording's transcript: the voice continues the
// recording (closest likeness, keeps its accent). NULL: speaker fingerprint only (native
// pronunciation). 0 on success, 1 if cb stopped it, negative on error.
MX_API int          mx_voice_say(mx_voice * v, const char * text, const char * lang, int max_frames,
                                 const char * ref_text, mx_voice_chunk_cb cb, void * user);
MX_API void         mx_voice_free(mx_voice * v);
MX_API const char * mx_voice_last_error(void);

#ifdef __cplusplus
}
#endif
