/* xrt.h — C API for XENO Runtime (xrt)
 *
 * Link with: -lxrt_capi (or xrt_capi.lib on Windows)
 *
 * All functions return 0 on success, -1 on error.
 * Call xrt_last_error() for the error message after a failure.
 */
#ifndef XRT_H
#define XRT_H

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles */
typedef struct XrtRuntime XrtRuntime;
typedef struct XrtSession XrtSession;

/* Error handling — returns thread-local error string, valid until next error */
const char *xrt_last_error(void);

/* Runtime — model loading */
int xrt_runtime_load(const char *model_path, XrtRuntime **out);
const char *xrt_runtime_model_name(const XrtRuntime *rt);
void xrt_runtime_free(XrtRuntime *rt);

/* Session — inference context */
int xrt_session_new(const XrtRuntime *rt, XrtSession **out);
void xrt_session_reset(XrtSession *sess);
void xrt_session_free(XrtSession *sess);

/* Generation */
int xrt_generate(XrtSession *sess, const char *prompt,
                 unsigned int max_tokens, float temperature,
                 char **out);
int xrt_generate_stream(XrtSession *sess, const char *prompt,
                        unsigned int max_tokens, float temperature,
                        void (*callback)(void *user_data, const char *piece),
                        void *user_data);

/* Memory management */
void xrt_string_free(char *s);

#ifdef __cplusplus
}
#endif

#endif /* XRT_H */
