#define _GNU_SOURCE

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef enum {
    MODE_COMPUTE = 0,
    MODE_MEMORY = 1,
    MODE_CACHE = 2,
    MODE_BRANCH = 3,
    MODE_FP = 4,
    MODE_MIXED = 5,
} workload_mode_t;

typedef struct {
    atomic_int *stop_flag;
    atomic_int *phase_index;
    workload_mode_t mode;
    size_t memory_span;
    uint64_t seed;
} worker_args_t;

static volatile double global_sink_double = 0.0;
static volatile uint64_t global_sink_u64 = 0;
static const int COMPUTE_CHUNK = 2000;
static const int MEMORY_CHUNK = 4096;
static const int CACHE_CHUNK = 2048;
static const int BRANCH_CHUNK = 4000;
static const int FP_CHUNK = 2000;

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static int stop_requested(atomic_int *stop_flag) {
    return atomic_load(stop_flag) != 0;
}

static void run_compute(uint64_t *state, atomic_int *stop_flag) {
    uint64_t acc = *state;
    for (int base = 0; base < 50000 && !stop_requested(stop_flag); base += COMPUTE_CHUNK) {
        int limit = base + COMPUTE_CHUNK;
        if (limit > 50000) {
            limit = 50000;
        }
        for (int i = base; i < limit; ++i) {
            acc = (acc * 1664525ull) + 1013904223ull;
            acc ^= acc >> 7;
            acc += (uint64_t)i * 2654435761ull;
        }
    }
    *state = acc;
    global_sink_u64 ^= acc;
}

static void run_memory(uint64_t *state, uint64_t *buffer, size_t count, atomic_int *stop_flag) {
    uint64_t acc = *state;
    size_t chunk_stride = (size_t)MEMORY_CHUNK * 8u;
    for (size_t base = 0; base < count && !stop_requested(stop_flag); base += chunk_stride) {
        size_t limit = base + chunk_stride;
        if (limit > count) {
            limit = count;
        }
        for (size_t i = base; i < limit; i += 8) {
            buffer[i] += acc + i;
            acc += buffer[i];
        }
    }
    *state = acc;
    global_sink_u64 ^= acc;
}

static void run_cache(uint64_t *state, uint64_t *buffer, size_t count, atomic_int *stop_flag) {
    uint64_t acc = *state;
    for (int base = 0; base < 60000 && !stop_requested(stop_flag); base += CACHE_CHUNK) {
        int limit = base + CACHE_CHUNK;
        if (limit > 60000) {
            limit = 60000;
        }
        for (int i = base; i < limit; ++i) {
            size_t idx = (size_t)(xorshift64(state) % count);
            buffer[idx] ^= acc + idx;
            acc += buffer[idx];
        }
    }
    *state = acc;
    global_sink_u64 ^= acc;
}

static void run_branch(uint64_t *state, atomic_int *stop_flag) {
    uint64_t acc = *state;
    for (int base = 0; base < 100000 && !stop_requested(stop_flag); base += BRANCH_CHUNK) {
        int limit = base + BRANCH_CHUNK;
        if (limit > 100000) {
            limit = 100000;
        }
        for (int i = base; i < limit; ++i) {
            uint64_t next = xorshift64(state);
            if (next & 1ull) {
                acc += next * 3ull;
            } else if (next & 2ull) {
                acc ^= next >> 1;
            } else {
                acc -= next * 7ull;
            }
        }
    }
    *state = acc;
    global_sink_u64 ^= acc;
}

static void run_fp(uint64_t *state, atomic_int *stop_flag) {
    double x = (double)(*state % 1000ull) + 1.0;
    double y = 0.999991;
    for (int base = 0; base < 50000 && !stop_requested(stop_flag); base += FP_CHUNK) {
        int limit = base + FP_CHUNK;
        if (limit > 50000) {
            limit = 50000;
        }
        for (int i = base; i < limit; ++i) {
            x = x * y + sqrt(x + 1.0);
            y += 0.0000001;
            if (y > 1.0002) {
                y = 0.999991;
            }
        }
    }
    *state += (uint64_t)x;
    global_sink_double += x;
}

static void run_mixed(uint64_t *state, uint64_t *buffer, size_t count, int phase, atomic_int *stop_flag) {
    switch (phase % 5) {
        case 0:
            run_compute(state, stop_flag);
            break;
        case 1:
            run_memory(state, buffer, count, stop_flag);
            break;
        case 2:
            run_branch(state, stop_flag);
            break;
        case 3:
            run_fp(state, stop_flag);
            break;
        default:
            run_cache(state, buffer, count, stop_flag);
            break;
    }
}

static void *worker_main(void *opaque) {
    worker_args_t *args = (worker_args_t *)opaque;
    size_t count = args->memory_span / sizeof(uint64_t);
    if (count < 1024) {
        count = 1024;
    }
    uint64_t *buffer = calloc(count, sizeof(uint64_t));
    if (buffer == NULL) {
        perror("calloc");
        return NULL;
    }
    uint64_t state = args->seed;
    while (!atomic_load(args->stop_flag)) {
        if (args->mode == MODE_COMPUTE) {
            run_compute(&state, args->stop_flag);
        } else if (args->mode == MODE_MEMORY) {
            run_memory(&state, buffer, count, args->stop_flag);
        } else if (args->mode == MODE_CACHE) {
            run_cache(&state, buffer, count, args->stop_flag);
        } else if (args->mode == MODE_BRANCH) {
            run_branch(&state, args->stop_flag);
        } else if (args->mode == MODE_FP) {
            run_fp(&state, args->stop_flag);
        } else {
            run_mixed(&state, buffer, count, atomic_load(args->phase_index), args->stop_flag);
        }
    }
    free(buffer);
    return NULL;
}

static workload_mode_t parse_mode(const char *value) {
    if (strcmp(value, "compute") == 0) return MODE_COMPUTE;
    if (strcmp(value, "memory") == 0) return MODE_MEMORY;
    if (strcmp(value, "cache") == 0) return MODE_CACHE;
    if (strcmp(value, "branch") == 0) return MODE_BRANCH;
    if (strcmp(value, "fp") == 0) return MODE_FP;
    return MODE_MIXED;
}

static const char *phase_name(int phase) {
    switch (phase % 5) {
        case 0: return "compute";
        case 1: return "memory";
        case 2: return "branch";
        case 3: return "fp";
        default: return "mixed";
    }
}

int main(int argc, char **argv) {
    const char *workload = "compute";
    int threads = 1;
    int duration_ms = 1500;
    int phase_ms = 150;
    const char *phase_log_path = NULL;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--workload") == 0 && i + 1 < argc) {
            workload = argv[++i];
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--duration-ms") == 0 && i + 1 < argc) {
            duration_ms = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--phase-ms") == 0 && i + 1 < argc) {
            phase_ms = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--phase-log") == 0 && i + 1 < argc) {
            phase_log_path = argv[++i];
        }
    }

    atomic_int stop_flag = 0;
    atomic_int phase_index = 0;
    workload_mode_t mode = parse_mode(workload);

    pthread_t *worker_threads = calloc((size_t)threads, sizeof(pthread_t));
    worker_args_t *worker_args = calloc((size_t)threads, sizeof(worker_args_t));
    if (worker_threads == NULL || worker_args == NULL) {
        perror("calloc");
        return 1;
    }

    uint64_t start_ns = now_ns();
    for (int i = 0; i < threads; ++i) {
        worker_args[i].stop_flag = &stop_flag;
        worker_args[i].phase_index = &phase_index;
        worker_args[i].mode = mode;
        worker_args[i].memory_span = (mode == MODE_MEMORY) ? (128ull << 20) : (mode == MODE_CACHE ? (256ull << 20) : (64ull << 20));
        worker_args[i].seed = (uint64_t)(i + 1) * 0x9e3779b97f4a7c15ull;
        if (pthread_create(&worker_threads[i], NULL, worker_main, &worker_args[i]) != 0) {
            perror("pthread_create");
            return 1;
        }
    }

    FILE *phase_log = NULL;
    if (phase_log_path != NULL) {
        phase_log = fopen(phase_log_path, "w");
        if (phase_log != NULL) {
            fprintf(phase_log, "start_ms,end_ms,phase_label\n");
        }
    }

    int previous_phase = -1;
    uint64_t end_ns = start_ns + (uint64_t)duration_ms * 1000000ull;
    while (now_ns() < end_ns) {
        uint64_t elapsed_ms = (now_ns() - start_ns) / 1000000ull;
        if (mode == MODE_MIXED) {
            int current_phase = (int)(elapsed_ms / (uint64_t)phase_ms);
            atomic_store(&phase_index, current_phase);
            if (phase_log != NULL && current_phase != previous_phase) {
                if (previous_phase >= 0) {
                    uint64_t phase_start = (uint64_t)previous_phase * (uint64_t)phase_ms;
                    uint64_t phase_end = elapsed_ms;
                    fprintf(phase_log, "%llu,%llu,%s\n",
                            (unsigned long long)phase_start,
                            (unsigned long long)phase_end,
                            phase_name(previous_phase));
                }
                previous_phase = current_phase;
            }
        }
        usleep(1000);
    }

    atomic_store(&stop_flag, 1);
    for (int i = 0; i < threads; ++i) {
        pthread_join(worker_threads[i], NULL);
    }

    if (phase_log != NULL) {
        if (mode == MODE_MIXED && previous_phase >= 0) {
            uint64_t total_ms = (now_ns() - start_ns) / 1000000ull;
            uint64_t phase_start = (uint64_t)previous_phase * (uint64_t)phase_ms;
            fprintf(phase_log, "%llu,%llu,%s\n",
                    (unsigned long long)phase_start,
                    (unsigned long long)total_ms,
                    phase_name(previous_phase));
        }
        fclose(phase_log);
    }

    free(worker_threads);
    free(worker_args);
    fprintf(stderr, "sink_u64=%llu sink_double=%.4f\n",
            (unsigned long long)global_sink_u64, global_sink_double);
    return 0;
}
