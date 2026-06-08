#include "ds4.h"

/* Purpose-built throughput benchmark.
 *
 * The benchmark walks one fixed token sequence to configurable context
 * frontiers, measuring only the newest prefill interval at each frontier.  It
 * then snapshots the live session in memory, performs a fixed greedy decode
 * run without allowing EOS, restores the snapshot, and continues to the next
 * frontier.  Snapshot save/restore time is intentionally outside both timing
 * windows.
 */

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(_WIN32) && !defined(__MINGW32__)
/* Native Windows GPU (HIP/clang-MSVC) build: MSVC lacks clock_gettime/
 * CLOCK_MONOTONIC; the shim supplies them. The MinGW CPU build already has
 * them via <time.h>, so it does not include the shim. See win/ds4_win.h. */
#include "ds4_win.h"
#endif

typedef struct {
    const char *model_path;
    const char *prompt_path;
    const char *chat_prompt_path;
    const char *system;
    const char *csv_path;
    const char *save_session_path;
    const char *load_session_path;
    ds4_backend backend;
    int threads;
    int ctx_start;
    int ctx_max;
    int ctx_alloc;
    int step_incr;
    int gen_tokens;
    double step_mul;
    bool warm_weights;
    bool quality;
} bench_config;

static double bench_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static void usage(FILE *fp) {
    fprintf(fp,
        "Usage: ds4-bench --prompt-file FILE [options]\n"
        "\n"
        "Benchmarks instantaneous prefill and generation throughput at context\n"
        "frontiers such as 2048, 4096, 6144, ... . Generation is always greedy,\n"
        "runs for exactly --gen-tokens tokens, and skips EOS so every row is\n"
        "comparable.\n"
        "\n"
        "Input:\n"
        "  --prompt-file FILE\n"
        "      Raw benchmark text. The fixed token sequence is sliced at each frontier.\n"
        "  --chat-prompt-file FILE\n"
        "      Render FILE as one no-thinking chat user message, then slice that sequence.\n"
        "  -sys, --system TEXT\n"
        "      System prompt used only with --chat-prompt-file.\n"
        "\n"
        "Model and backend:\n"
        "  -m, --model FILE       GGUF model path. Default: ds4flash.gguf\n"
        "  --metal | --cuda | --cpu | --backend NAME\n"
        "      Select backend explicitly. Defaults to Metal on macOS, CUDA elsewhere.\n"
        "  -t, --threads N        CPU helper threads.\n"
        "  --quality              Prefer exact kernels where applicable.\n"
        "  --warm-weights         Touch mapped tensor pages before benchmarking.\n"
        "\n"
        "Sweep:\n"
        "  --ctx-start N          First measured frontier. Default: 2048\n"
        "  --ctx-max N            Last measured frontier. Default: 32768\n"
        "  --ctx-alloc N          Allocated context. Default: ctx-max + gen-tokens + 1\n"
        "  --step-mul F           Multiplicative step. Default: 1\n"
        "  --step-incr N          Linear step when --step-mul is 1. Default: 2048\n"
        "  --gen-tokens N         Greedy decode tokens per frontier. Default: 128\n"
        "\n"
        "Output:\n"
        "  --csv FILE             Write CSV there instead of stdout.\n"
        "  -h, --help             Show this help.\n"
        "\n"
        "Prefix-cache demo (default off; both run a single frontier = --ctx-start):\n"
        "  --save-session FILE\n"
        "      Cold-prefill the prefix, then write the serialized session payload\n"
        "      (ds4_session_save_payload) to FILE. Reports cold prefill time and\n"
        "      payload bytes. Use this to populate a prefix-cache entry.\n"
        "  --load-session FILE\n"
        "      Restore the session payload from FILE (ds4_session_load_payload)\n"
        "      instead of cold-prefilling. Reports restore (warm) time and the\n"
        "      prefill time it skipped. The restored prefix must match the prompt\n"
        "      sliced at --ctx-start. Combine with DS4_BENCH_DUMP_GEN=1 to verify\n"
        "      the continued greedy generation matches a cold run.\n");
}

static int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT_MAX) {
        fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static double parse_double_arg(const char *s, const char *opt) {
    char *end = NULL;
    double v = strtod(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v)) {
        fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-bench: %s requires an argument\n", opt);
        exit(2);
    }
    return argv[++*i];
}

static ds4_backend parse_backend(const char *s, const char *opt) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
    fprintf(stderr, "ds4-bench: valid backends are: metal, cuda, cpu\n");
    exit(2);
}

static ds4_backend default_backend(void) {
#ifdef DS4_NO_GPU
    return DS4_BACKEND_CPU;
#elif defined(__APPLE__)
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4-bench: failed to open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "ds4-bench: failed to seek %s\n", path);
        fclose(fp);
        exit(1);
    }
    long n = ftell(fp);
    if (n < 0) {
        fprintf(stderr, "ds4-bench: failed to tell %s\n", path);
        fclose(fp);
        exit(1);
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "ds4-bench: failed to rewind %s\n", path);
        fclose(fp);
        exit(1);
    }
    char *buf = malloc((size_t)n + 1);
    if (!buf) {
        fprintf(stderr, "ds4-bench: out of memory reading %s\n", path);
        fclose(fp);
        exit(1);
    }
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) {
        fprintf(stderr, "ds4-bench: failed to read %s\n", path);
        free(buf);
        fclose(fp);
        exit(1);
    }
    fclose(fp);
    buf[n] = '\0';
    return buf;
}

static bench_config parse_options(int argc, char **argv) {
    bench_config c = {
        .model_path = "ds4flash.gguf",
        .system = "You are a helpful assistant.",
        .backend = default_backend(),
        .ctx_start = 2048,
        .ctx_max = 32768,
        .step_incr = 2048,
        .gen_tokens = 128,
        .step_mul = 1.0,
    };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prompt-file")) {
            c.prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--chat-prompt-file")) {
            c.chat_prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-sys") || !strcmp(arg, "--system")) {
            c.system = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--ctx-start")) {
            c.ctx_start = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--ctx-max")) {
            c.ctx_max = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--ctx-alloc")) {
            c.ctx_alloc = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--step-incr")) {
            c.step_incr = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--step-mul")) {
            c.step_mul = parse_double_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--gen-tokens") || !strcmp(arg, "--tokens") || !strcmp(arg, "-n")) {
            c.gen_tokens = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--csv")) {
            c.csv_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--save-session")) {
            c.save_session_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--load-session")) {
            c.load_session_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.threads = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--backend")) {
            c.backend = parse_backend(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--metal")) {
            c.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--cuda")) {
            c.backend = DS4_BACKEND_CUDA;
        } else if (!strcmp(arg, "--cpu")) {
            c.backend = DS4_BACKEND_CPU;
        } else if (!strcmp(arg, "--quality")) {
            c.quality = true;
        } else if (!strcmp(arg, "--warm-weights")) {
            c.warm_weights = true;
        } else {
            fprintf(stderr, "ds4-bench: unknown option: %s\n", arg);
            usage(stderr);
            exit(2);
        }
    }

    if (!!c.prompt_path == !!c.chat_prompt_path) {
        fprintf(stderr, "ds4-bench: specify exactly one of --prompt-file or --chat-prompt-file\n");
        exit(2);
    }
    if (c.save_session_path && c.load_session_path) {
        fprintf(stderr, "ds4-bench: use only one of --save-session / --load-session per run\n");
        exit(2);
    }
    if (c.ctx_start > c.ctx_max) {
        fprintf(stderr, "ds4-bench: --ctx-start must be <= --ctx-max\n");
        exit(2);
    }
    if (c.step_mul < 1.0) {
        fprintf(stderr, "ds4-bench: --step-mul must be >= 1\n");
        exit(2);
    }
    if (c.step_mul == 1.0 && c.step_incr <= 0) {
        fprintf(stderr, "ds4-bench: --step-incr must be positive when --step-mul is 1\n");
        exit(2);
    }
    if (c.ctx_max > INT_MAX - c.gen_tokens - 1) {
        fprintf(stderr, "ds4-bench: requested context is too large\n");
        exit(2);
    }
    if (c.ctx_alloc == 0) c.ctx_alloc = c.ctx_max + c.gen_tokens + 1;
    if (c.ctx_alloc <= c.ctx_max + c.gen_tokens) {
        fprintf(stderr, "ds4-bench: --ctx-alloc must be greater than ctx-max + gen-tokens\n");
        exit(2);
    }
    return c;
}

static int next_frontier(const bench_config *c, int cur) {
    if (cur >= c->ctx_max) return c->ctx_max;
    int next;
    if (c->step_mul == 1.0) {
        if (cur > INT_MAX - c->step_incr) next = c->ctx_max;
        else next = cur + c->step_incr;
    } else {
        const double v = ceil((double)cur * c->step_mul);
        next = v > (double)INT_MAX ? c->ctx_max : (int)v;
        if (next <= cur) next = cur + 1;
    }
    if (next > c->ctx_max) next = c->ctx_max;
    return next;
}

static void log_context_memory(ds4_backend backend, int ctx_size) {
    ds4_context_memory m = ds4_context_memory_estimate(backend, ctx_size);
    fprintf(stderr,
            "ds4-bench: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)\n",
            (double)m.total_bytes / (1024.0 * 1024.0),
            ctx_size,
            ds4_backend_name(backend),
            m.prefill_cap,
            m.raw_cap,
            m.comp_cap);
}

/* Greedy, EOS-skipping decode of cfg.gen_tokens tokens from the current session
 * frontier.  Used by both the sweep and the prefix-cache demo so the warm and
 * cold paths produce byte-identical generation when their prefixes match.  When
 * dump_gen is set, the decoded text is written to stderr for correctness checks. */
static int bench_decode_gen(ds4_engine *engine, ds4_session *session,
                            const bench_config *cfg, int eos, bool dump_gen,
                            double *out_sec) {
    char err[256];
    const double t0 = bench_now_sec();
    for (int i = 0; i < cfg->gen_tokens; i++) {
        if (ds4_session_pos(session) + 1 >= ds4_session_ctx(session)) {
            fprintf(stderr, "ds4-bench: generation would exceed allocated context\n");
            return 1;
        }
        const int token = ds4_session_argmax_excluding(session, eos);
        if (token < 0) {
            fprintf(stderr, "ds4-bench: failed to choose non-EOS token\n");
            return 1;
        }
        if (dump_gen) {
            size_t tlen = 0;
            char *piece = ds4_token_text(engine, token, &tlen);
            if (piece) { fwrite(piece, 1, tlen, stderr); free(piece); }
        }
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4-bench: decode failed: %s\n", err);
            return 1;
        }
    }
    if (out_sec) *out_sec = bench_now_sec() - t0;
    return 0;
}

/* Prefix-cache demonstration mode (default off).  Exercises the exact disk
 * payload API the server's KV disk cache uses (ds4_session_save_payload /
 * ds4_session_load_payload) so the orchestrator can measure cold prefill versus
 * a warm restore that eliminates it.  Runs a single frontier = cfg.ctx_start. */
static int run_session_file_mode(ds4_engine *engine, ds4_session *session,
                                 const ds4_tokens *prompt, const bench_config *cfg,
                                 int eos, bool dump_gen) {
    char err[256];
    const int frontier = cfg->ctx_start;
    ds4_tokens prefix = { .v = prompt->v, .len = frontier, .cap = frontier };

    if (cfg->load_session_path) {
        FILE *fp = fopen(cfg->load_session_path, "rb");
        if (!fp) {
            fprintf(stderr, "ds4-bench: failed to open %s: %s\n",
                    cfg->load_session_path, strerror(errno));
            return 1;
        }
        if (fseek(fp, 0, SEEK_END) != 0) { fprintf(stderr, "ds4-bench: seek failed\n"); fclose(fp); return 1; }
        long fsz = ftell(fp);
        if (fsz < 0 || fseek(fp, 0, SEEK_SET) != 0) { fprintf(stderr, "ds4-bench: tell/rewind failed\n"); fclose(fp); return 1; }

        const double t0 = bench_now_sec();
        const int rc = ds4_session_load_payload(session, fp, (uint64_t)fsz, err, sizeof(err));
        fclose(fp);
        const double restore_sec = bench_now_sec() - t0;
        if (rc != 0) {
            fprintf(stderr, "ds4-bench: load-session failed: %s\n", err);
            return 1;
        }
        /* The restored checkpoint must equal the prompt sliced at the frontier,
         * otherwise the continued generation would diverge from a cold run. */
        const ds4_tokens *live = ds4_session_tokens(session);
        if (live->len != frontier) {
            fprintf(stderr, "ds4-bench: restored prefix is %d tokens, expected frontier %d\n",
                    live->len, frontier);
            return 1;
        }
        for (int i = 0; i < frontier; i++) {
            if (live->v[i] != prompt->v[i]) {
                fprintf(stderr, "ds4-bench: restored prefix diverges from prompt at token %d\n", i);
                return 1;
            }
        }
        fprintf(stderr,
                "ds4-bench: WARM restore of %d-token prefix from %s in %.3f s "
                "(%.1f MiB, %.0f MiB/s) -- cold prefill SKIPPED\n",
                frontier, cfg->load_session_path, restore_sec,
                (double)fsz / (1024.0 * 1024.0),
                restore_sec > 0.0 ? ((double)fsz / (1024.0 * 1024.0)) / restore_sec : 0.0);

        if (dump_gen) fprintf(stderr, "ds4-bench: gen@%d(warm): ", frontier);
        double gen_sec = 0.0;
        if (bench_decode_gen(engine, session, cfg, eos, dump_gen, &gen_sec) != 0) return 1;
        if (dump_gen) { fputc('\n', stderr); fflush(stderr); }
        printf("mode,frontier,restore_sec,payload_bytes,gen_tokens,gen_tps\n");
        printf("warm,%d,%.4f,%lld,%d,%.2f\n", frontier, restore_sec,
               (long long)fsz, cfg->gen_tokens,
               gen_sec > 0.0 ? (double)cfg->gen_tokens / gen_sec : 0.0);
        return 0;
    }

    /* --save-session: cold prefill, save payload, then generate (so the saver's
     * own continued output can be compared against a later warm run). */
    const double prefill_t0 = bench_now_sec();
    if (ds4_session_sync(session, &prefix, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-bench: cold prefill to %d failed: %s\n", frontier, err);
        return 1;
    }
    const double prefill_sec = bench_now_sec() - prefill_t0;

    FILE *fp = fopen(cfg->save_session_path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4-bench: failed to open %s: %s\n",
                cfg->save_session_path, strerror(errno));
        return 1;
    }
    const uint64_t payload_bytes = ds4_session_payload_bytes(session);
    const double save_t0 = bench_now_sec();
    const int rc = ds4_session_save_payload(session, fp, err, sizeof(err));
    const int closerc = fclose(fp);
    const double save_sec = bench_now_sec() - save_t0;
    if (rc != 0 || closerc != 0) {
        fprintf(stderr, "ds4-bench: save-session failed: %s\n", rc != 0 ? err : "fclose");
        return 1;
    }
    fprintf(stderr,
            "ds4-bench: COLD prefill of %d tokens in %.3f s (%.1f tok/s); "
            "saved %.1f MiB payload to %s in %.3f s\n",
            frontier, prefill_sec, prefill_sec > 0.0 ? frontier / prefill_sec : 0.0,
            (double)payload_bytes / (1024.0 * 1024.0), cfg->save_session_path, save_sec);

    if (dump_gen) fprintf(stderr, "ds4-bench: gen@%d(cold): ", frontier);
    double gen_sec = 0.0;
    if (bench_decode_gen(engine, session, cfg, eos, dump_gen, &gen_sec) != 0) return 1;
    if (dump_gen) { fputc('\n', stderr); fflush(stderr); }
    printf("mode,frontier,prefill_sec,prefill_tps,payload_bytes,gen_tokens,gen_tps\n");
    printf("cold,%d,%.4f,%.2f,%llu,%d,%.2f\n", frontier, prefill_sec,
           prefill_sec > 0.0 ? (double)frontier / prefill_sec : 0.0,
           (unsigned long long)payload_bytes, cfg->gen_tokens,
           gen_sec > 0.0 ? (double)cfg->gen_tokens / gen_sec : 0.0);
    return 0;
}

int main(int argc, char **argv) {
    bench_config cfg = parse_options(argc, argv);
    log_context_memory(cfg.backend, cfg.ctx_alloc);

    ds4_engine_options opt = {
        .model_path = cfg.model_path,
        .backend = cfg.backend,
        .n_threads = cfg.threads,
        .warm_weights = cfg.warm_weights,
        .quality = cfg.quality,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) return 1;

    char *text = read_file(cfg.prompt_path ? cfg.prompt_path : cfg.chat_prompt_path);
    ds4_tokens prompt = {0};
    if (cfg.chat_prompt_path) {
        ds4_encode_chat_prompt(engine, cfg.system, text, DS4_THINK_NONE, &prompt);
    } else {
        ds4_tokenize_text(engine, text, &prompt);
    }
    free(text);

    if (prompt.len < cfg.ctx_max) {
        fprintf(stderr,
                "ds4-bench: prompt has %d tokens, need at least --ctx-max=%d\n",
                prompt.len,
                cfg.ctx_max);
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg.ctx_alloc) != 0) {
        fprintf(stderr, "ds4-bench: failed to create session\n");
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }

    /* DS4_BENCH_DUMP_GEN: dump greedy generation to stderr for correctness checks. */
    const bool dump_gen_env = getenv("DS4_BENCH_DUMP_GEN") != NULL &&
                              getenv("DS4_BENCH_DUMP_GEN")[0] != '\0' &&
                              getenv("DS4_BENCH_DUMP_GEN")[0] != '0';

    /* Prefix-cache demo path (default off).  Runs a single frontier and exits. */
    if (cfg.save_session_path || cfg.load_session_path) {
        const int srcrc = run_session_file_mode(engine, session, &prompt, &cfg,
                                                 ds4_token_eos(engine), dump_gen_env);
        ds4_session_free(session);
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return srcrc;
    }

    FILE *out = stdout;
    if (cfg.csv_path) {
        out = fopen(cfg.csv_path, "wb");
        if (!out) {
            fprintf(stderr, "ds4-bench: failed to open %s: %s\n", cfg.csv_path, strerror(errno));
            ds4_session_free(session);
            ds4_tokens_free(&prompt);
            ds4_engine_close(engine);
            return 1;
        }
    }
    fprintf(out, "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes\n");
    fflush(out);

    const int eos = ds4_token_eos(engine);
    const bool dump_gen = dump_gen_env;
    ds4_session_snapshot snap = {0};
    char err[256];
    int previous = 0;
    int rc = 0;

    for (int frontier = cfg.ctx_start; ; frontier = next_frontier(&cfg, frontier)) {
        ds4_tokens prefix = {
            .v = prompt.v,
            .len = frontier,
            .cap = frontier,
        };

        const double prefill_t0 = bench_now_sec();
        if (ds4_session_sync(session, &prefix, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4-bench: prefill to %d failed: %s\n", frontier, err);
            rc = 1;
            break;
        }
        const double prefill_t1 = bench_now_sec();
        const double prefill_sec = prefill_t1 - prefill_t0;
        const int prefill_tokens = frontier - previous;

        if (ds4_session_save_snapshot(session, &snap, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4-bench: snapshot at %d failed: %s\n", frontier, err);
            rc = 1;
            break;
        }

        if (dump_gen) fprintf(stderr, "ds4-bench: gen@%d: ", frontier);
        const double gen_t0 = bench_now_sec();
        for (int i = 0; i < cfg.gen_tokens; i++) {
            if (ds4_session_pos(session) + 1 >= ds4_session_ctx(session)) {
                fprintf(stderr, "ds4-bench: generation would exceed allocated context at frontier %d\n", frontier);
                rc = 1;
                break;
            }
            const int token = ds4_session_argmax_excluding(session, eos);
            if (token < 0) {
                fprintf(stderr, "ds4-bench: failed to choose non-EOS token at frontier %d\n", frontier);
                rc = 1;
                break;
            }
            if (dump_gen) {
                size_t tlen = 0;
                char *piece = ds4_token_text(engine, token, &tlen);
                if (piece) {
                    fwrite(piece, 1, tlen, stderr);
                    free(piece);
                }
            }
            if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: decode at frontier %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
        }
        const double gen_t1 = bench_now_sec();
        if (dump_gen) { fputc('\n', stderr); fflush(stderr); }
        if (rc != 0) break;

        if (ds4_session_load_snapshot(session, &snap, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4-bench: restore at %d failed: %s\n", frontier, err);
            rc = 1;
            break;
        }

        const double gen_sec = gen_t1 - gen_t0;
        fprintf(out,
                "%d,%d,%.2f,%d,%.2f,%llu\n",
                frontier,
                prefill_tokens,
                prefill_sec > 0.0 ? (double)prefill_tokens / prefill_sec : 0.0,
                cfg.gen_tokens,
                gen_sec > 0.0 ? (double)cfg.gen_tokens / gen_sec : 0.0,
                (unsigned long long)snap.len);
        fflush(out);

        previous = frontier;
        if (frontier >= cfg.ctx_max) break;
    }

    if (out != stdout) fclose(out);
    ds4_session_snapshot_free(&snap);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    return rc;
}
