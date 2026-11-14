// mentalese's C ABI over llama.cpp + mtmd: load a GGUF model (optionally with an audio
// projector) and generate greedily from a ready-made prompt. One self-contained shared library
// per GPU backend; llama.cpp and its ggml are linked in statically and kept private, so this
// can share a process with the voice engine's own (different) ggml.
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

#define MX_LLM_ABI 1

typedef struct mx_llm mx_llm;
typedef void (*mx_log_cb)(int level, const char * msg);  // level: 2 info, 3 warn, 4 error

MX_API int          mx_llm_abi(void);
MX_API void         mx_llm_log(mx_log_cb cb);  // NULL: silent
// GPU devices as lines "name\tdescription\tfree_bytes\ttotal_bytes"; valid until the next call.
MX_API const char * mx_llm_devices(void);
// The marker an audio prompt must contain where the audio goes.
MX_API const char * mx_llm_marker(void);

// mmproj may be NULL (text only). mmproj_gpu=0 keeps the audio encoder on the CPU.
MX_API mx_llm *     mx_llm_load(const char * model, const char * mmproj, int mmproj_gpu,
                                int n_ctx, int n_batch, int n_ubatch, int n_threads);
// Greedy generation from `prompt` (special tokens parsed). With audio (16 kHz mono float),
// the prompt must contain mx_llm_marker(). Returns UTF-8 text valid until the next call on
// this handle, or NULL on error. Not thread-safe per handle.
MX_API const char * mx_llm_generate(mx_llm * m, const char * prompt, const float * audio, int n_audio,
                                    int max_tokens);
MX_API void         mx_llm_free(mx_llm * m);
// Last error on the calling thread.
MX_API const char * mx_llm_last_error(void);

#ifdef __cplusplus
}
#endif
