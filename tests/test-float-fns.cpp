#include "ggml-cpu.h"
#include "ggml.h"
#include "vec.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define DOT_F16_ERROR_THRESHOLD   1e-2f
#define DOT_BF16_ERROR_THRESHOLD  1e-2f
#define MAD_F16_ERROR_THRESHOLD   4e-3f
#define SCALE_F16_ERROR_THRESHOLD 4e-3f
#define DOT_F16_UNROLL_THRESHOLD  1e-2f

#define CPU_BF16_TO_FP32_THRESHOLD 0
#define CPU_FP16_TO_FP32_THRESHOLD 0

#define SILU_F32_THRESHOLD 1e-4f

// Utilities for conversion
template <typename T> float to_float(T x);
template <> float to_float(float x) {
    return x;
}
template <> float to_float(ggml_fp16_t x) {
    return ggml_fp16_to_fp32(x);
}
template <> float to_float(ggml_bf16_t x) {
    return ggml_bf16_to_fp32(x);
}

// Helper to convert Float -> Any Type
template <typename T> T from_float(float x);
template <> float from_float(float x) {
    return x;
}

template <> ggml_fp16_t from_float(float x) {
    return ggml_fp32_to_fp16(x);
}

template <> ggml_bf16_t from_float(float x) {
    return ggml_fp32_to_bf16(x);
}

static std::mt19937 g_rng;

enum DataPattern {
    RANDOM,
    COSINE,
    SPECIAL,  // mix of NAN,INF, 0
    FILL      // Constant value
};

template <typename T> std::vector<T> generate_data(size_t n, DataPattern pattern, float fill_value = 0.0f) {
    std::vector<T>                        data(n);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < n; ++i) {
        float val = 0.0f;
        switch (pattern) {
            case RANDOM:
                val = dist(g_rng);
                break;
            case COSINE:
                val = 2.0f * std::cos(i * 0.1f) - 0.1f;
                break;
            case FILL:
                val = fill_value;
                break;
            case SPECIAL:
                if (i % 4 == 0) {
                    val = INFINITY;
                } else if (i % 4 == 1) {
                    val = -INFINITY;
                } else if (i % 4 == 2) {
                    val = NAN;
                } else {
                    val = 0.0f;
                }
                break;
        }
        data[i] = from_float<T>(val);
    }
    return data;
}

template <typename T>
bool check_error(size_t n, const T * ref, const T * opt, const char * func_name, float threshold) {
    bool passed = true;

    for (size_t i = 0; i < n; ++i) {
        float v_ref = to_float(ref[i]);
        float v_opt = to_float(opt[i]);
        // --- Rule A: Handle NaN ---
        bool ref_nan = std::isnan(v_ref);
        bool opt_nan = std::isnan(v_opt);

        if (ref_nan || opt_nan) {
            if (ref_nan && opt_nan) {
                continue;  // Both are NaN -> Pass
            }
            printf("[FAIL] %s (idx %zu): NaN mismatch. Ref: %f, Opt: %f\n", func_name, i, v_ref, v_opt);
            passed = false;
            break;
        }

        // --- Rule B: Handle Infinity ---
        bool ref_inf = std::isinf(v_ref);
        bool opt_inf = std::isinf(v_opt);

        if (ref_inf || opt_inf) {
            if (ref_inf && opt_inf) {
                if (std::signbit(v_ref) != std::signbit(v_opt)) {
                    printf("[FAIL] %s (idx %zu): Inf Sign mismatch. Ref: %f, Opt: %f\n", func_name, i, v_ref, v_opt);
                    passed = false;
                    break;
                }
                continue;
            }
            printf("[FAIL] %s (idx %zu): Inf mismatch. Ref: %f, Opt: %f\n", func_name, i, v_ref, v_opt);
            passed = false;
            break;
        }
        float err = std::fabs(v_ref - v_opt);
        if (err > threshold) {
            printf("[FAIL] %s (idx %zu): Error %g > Threshold %g (Ref: %f, Opt: %f)\n", func_name, i, err,
                   threshold, v_ref, v_opt);
            passed = false;
            break;
        }
    }

    return passed;
}

// reference code implementations (copied from the fallback scalar impl for each kernel)
static inline void ggml_vec_mad_f16_reference(const int                         n,
                                              ggml_fp16_t * GGML_RESTRICT       y,
                                              const ggml_fp16_t * GGML_RESTRICT x,
                                              const float                       v) {
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(y[i]) + ggml_fp16_to_fp32(x[i]) * v);
    }
}

static void ggml_vec_mad_f32_reference(const int                   n,
                                       float * GGML_RESTRICT       y,
                                       const float * GGML_RESTRICT x,
                                       const float                 v) {
    for (int i = 0; i < n; ++i) {
        y[i] += x[i] * v;
    }
}

static void ggml_vec_dot_bf16_reference(int                         n,
                                        float * GGML_RESTRICT       s,
                                        size_t                      bs,
                                        ggml_bf16_t * GGML_RESTRICT x,
                                        size_t                      bx,
                                        ggml_bf16_t * GGML_RESTRICT y,
                                        size_t                      by,
                                        int                         nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);
    int        i    = 0;
    ggml_float sumf = 0;
    for (; i < n; ++i) {
        sumf += (ggml_float) (ggml_bf16_to_fp32(x[i]) * ggml_bf16_to_fp32(y[i]));
    }
    *s = sumf;
}

static inline void ggml_vec_scale_f16_reference(const int n, ggml_fp16_t * y, const float v) {
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(y[i]) * v);
    }
}

static void ggml_vec_dot_f16_reference(int                         n,
                                       float * GGML_RESTRICT       s,
                                       size_t                      bs,
                                       ggml_fp16_t * GGML_RESTRICT x,
                                       size_t                      bx,
                                       ggml_fp16_t * GGML_RESTRICT y,
                                       size_t                      by,
                                       int                         nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    ggml_float sumf = 0.0;

    for (int i = 0; i < n; ++i) {
        sumf += (ggml_float) (ggml_fp16_to_fp32(x[i]) * ggml_fp16_to_fp32(y[i]));
    }

    *s = sumf;
}

inline static void ggml_vec_dot_f16_unroll_reference(const int                   n,
                                                     const int                   xs,
                                                     float * GGML_RESTRICT       s,
                                                     void * GGML_RESTRICT        xv,
                                                     ggml_fp16_t * GGML_RESTRICT y) {
    ggml_float sumf[GGML_VEC_DOT_UNROLL] = { 0.0 };

    ggml_fp16_t * GGML_RESTRICT x[GGML_VEC_DOT_UNROLL];

    for (int i = 0; i < GGML_VEC_DOT_UNROLL; ++i) {
        x[i] = (ggml_fp16_t *) ((char *) xv + i * xs);
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < GGML_VEC_DOT_UNROLL; ++j) {
            sumf[j] += (ggml_float) (ggml_fp16_to_fp32(x[j][i]) * ggml_fp16_to_fp32(y[i]));
        }
    }

    for (int i = 0; i < GGML_VEC_DOT_UNROLL; ++i) {
        s[i] = (float) sumf[i];
    }
}

static void ggml_vec_silu_f32_reference(const int n, float * y, const float * x) {
    int i = 0;
    for (; i < n; ++i) {
        y[i] = ggml_silu_f32(x[i]);
    }
}

static void ggml_cpu_bf16_to_fp32_reference(const ggml_bf16_t * x, float * y, int64_t n) {
    int i = 0;
    for (; i < n; i++) {
        y[i] = ggml_bf16_to_fp32(x[i]);
    }
}

static void ggml_cpu_fp16_to_fp32_reference(const ggml_fp16_t * x, float * y, int64_t n) {
    int i = 0;
    for (; i < n; i++) {
        y[i] = ggml_fp16_to_fp32(x[i]);
    }
}

// Test Functions

struct TestStats {
    int total_tests  = 0;
    int failed_tests = 0;

    void reset() {
        total_tests  = 0;
        failed_tests = 0;
    }
};

static void validate_dot_fp16(size_t n, std::vector<ggml_fp16_t> & x, std::vector<ggml_fp16_t> & y, TestStats & stats) {
    float s_ref = 0.0f;
    float s_opt = 0.0f;

    ggml_vec_dot_f16_reference(n, &s_ref, 0, x.data(), 0, y.data(), 0, 1);
    ggml_vec_dot_f16(n, &s_opt, 0, x.data(), 0, y.data(), 0, 1);

    stats.total_tests++;
    if (!check_error(1, &s_ref, &s_opt, "ggml_vec_dot_f16", DOT_F16_ERROR_THRESHOLD)) {
        stats.failed_tests++;
    }
}

static void validate_dot_bf16(size_t n, std::vector<ggml_bf16_t> & x, std::vector<ggml_bf16_t> & y, TestStats & stats) {
    float s_ref = 0.0f;
    float s_opt = 0.0f;

    ggml_vec_dot_bf16_reference(n, &s_ref, 0, x.data(), 0, y.data(), 0, 1);
    ggml_vec_dot_bf16(n, &s_opt, 0, x.data(), 0, y.data(), 0, 1);

    stats.total_tests++;
    if (!check_error(1, &s_ref, &s_opt, "ggml_vec_dot_bf16", DOT_BF16_ERROR_THRESHOLD)) {
        stats.failed_tests++;
    }
}

// SCALE
static void validate_scale_f16(size_t n, const std::vector<ggml_fp16_t> & y_in, float v, TestStats & stats) {
    std::vector<ggml_fp16_t> y_ref = y_in;
    std::vector<ggml_fp16_t> y_opt = y_in;

    ggml_vec_scale_f16_reference((int) n, y_ref.data(), v);
    ggml_vec_scale_f16((int) n, y_opt.data(), v);

    stats.total_tests++;
    if (!check_error(n, y_ref.data(), y_opt.data(), "ggml_vec_scale_f16", SCALE_F16_ERROR_THRESHOLD)) {
        stats.failed_tests++;
    }
}

static void validate_mad_f16(size_t                           n,
                             const std::vector<ggml_fp16_t> & y_in,
                             const std::vector<ggml_fp16_t> & x,
                             float                            v,
                             TestStats &                      stats) {
    std::vector<ggml_fp16_t> y_ref = y_in;
    std::vector<ggml_fp16_t> y_opt = y_in;

    ggml_vec_mad_f16_reference((int) n, y_ref.data(), x.data(), v);
    ggml_vec_mad_f16((int) n, y_opt.data(), x.data(), v);

    stats.total_tests++;
    if (!check_error(n, y_ref.data(), y_opt.data(), "ggml_vec_mad_f16", MAD_F16_ERROR_THRESHOLD)) {
        stats.failed_tests++;
    }
}

static void validate_dot_unroll(size_t n, int xs, void * xv, const std::vector<ggml_fp16_t> & y, TestStats & stats) {
    float s_ref[GGML_VEC_DOT_UNROLL];
    float s_opt[GGML_VEC_DOT_UNROLL];

    ggml_vec_dot_f16_unroll_reference((int) n, xs, s_ref, xv, (ggml_fp16_t *) y.data());
    ggml_vec_dot_f16_unroll((int) n, xs, s_opt, xv, (ggml_fp16_t *) y.data());

    stats.total_tests++;
    if (!check_error(GGML_VEC_DOT_UNROLL, s_ref, s_opt, "ggml_vec_dot_f16_unroll", DOT_F16_UNROLL_THRESHOLD)) {
        stats.failed_tests++;
    }
}

static void validate_conversion_bf16(size_t n, const std::vector<ggml_bf16_t> & x, TestStats & stats) {
    std::vector<float> y_ref(n);
    std::vector<float> y_opt(n);

    ggml_cpu_bf16_to_fp32_reference(x.data(), y_ref.data(), (int64_t) n);

    ggml_cpu_bf16_to_fp32(x.data(), y_opt.data(), (int64_t) n);

    stats.total_tests++;
    if (!check_error(n, y_ref.data(), y_opt.data(), "ggml_cpu_bf16_to_fp32", CPU_BF16_TO_FP32_THRESHOLD)) {
        stats.failed_tests++;
    }
}

static void validate_conversion_fp16(size_t n, const std::vector<ggml_fp16_t> & x, TestStats & stats) {
    std::vector<float> y_ref(n);
    std::vector<float> y_opt(n);

    ggml_cpu_fp16_to_fp32_reference(x.data(), y_ref.data(), (int64_t) n);

    ggml_cpu_fp16_to_fp32(x.data(), y_opt.data(), (int64_t) n);

    stats.total_tests++;
    if (!check_error(n, y_ref.data(), y_opt.data(), "ggml_cpu_fp16_to_fp32", CPU_FP16_TO_FP32_THRESHOLD)) {
        stats.failed_tests++;
    }
}

static void validate_silu(size_t n, const std::vector<float> & x, TestStats & stats) {
    std::vector<float> y_ref(n);
    std::vector<float> y_opt(n);

    ggml_vec_silu_f32_reference((int) n, y_ref.data(), x.data());
    ggml_vec_silu_f32((int) n, y_opt.data(), x.data());

    stats.total_tests++;
    if (!check_error(n, y_ref.data(), y_opt.data(), "ggml_vec_silu_f32", SILU_F32_THRESHOLD)) {
        stats.failed_tests++;
    }
}

/// ALL Cases tests for each kernel

static void test_vec_scale_f16_all_cases(TestStats & global_stats) {
    printf("\n--- Testing ggml_vec_scale_f16 ---\n");
    TestStats         local_stats;
    const size_t      sizes[]    = { 1, 7, 16, 31, 32, 64, 1024, 1025, 2048,8192 };
    const float       scalars[]  = { 0.0f, 1.0f, -1.0f, 3.14f, INFINITY, NAN };
    const DataPattern patterns[] = { COSINE, FILL, SPECIAL };

    for (size_t n : sizes) {
        for (DataPattern p : patterns) {
            auto y = generate_data<ggml_fp16_t>(n, p, 0.0f);
            for (float v : scalars) {
                validate_scale_f16(n, y, v, local_stats);
            }
        }
    }

    printf("Passed: %d/%d\n", local_stats.total_tests - local_stats.failed_tests, local_stats.total_tests);
    global_stats.total_tests += local_stats.total_tests;
    global_stats.failed_tests += local_stats.failed_tests;
}

static void test_vec_mad_f16_all_cases(TestStats & global_stats) {
    printf("\n--- Testing ggml_vec_mad_f16 ---\n");
    TestStats    local_stats;
    const size_t sizes[]   = { 1, 7, 16, 31, 32, 64, 1024, 1025, 2048,8192 };
    const float  scalars[] = { 0.0f, 1.0f, -1.0f, 2.71f, INFINITY, NAN };

    struct CaseConfig {
        DataPattern pX;
        DataPattern pY;
    };

    const CaseConfig configs[] = {
        { COSINE,  COSINE },
        { FILL,    COSINE },
        { COSINE,  FILL   },
        { SPECIAL, COSINE }
    };

    for (size_t n : sizes) {
        for (const auto & cfg : configs) {
            auto x = generate_data<ggml_fp16_t>(n, cfg.pX, 0.0f);
            auto y = generate_data<ggml_fp16_t>(n, cfg.pY, 0.0f);
            for (float v : scalars) {
                validate_mad_f16(n, y, x, v, local_stats);
            }
        }
    }

    printf("Passed: %d/%d\n", local_stats.total_tests - local_stats.failed_tests, local_stats.total_tests);
    global_stats.total_tests += local_stats.total_tests;
    global_stats.failed_tests += local_stats.failed_tests;
}

static void test_vec_dot_f16_all_cases(TestStats & global_stats) {
    printf("\n--- Testing ggml_vec_dot_f16 ---\n");
    TestStats    local_stats;
    const size_t sizes[] = { 1, 7, 16, 31, 32, 64, 1024, 1025, 2048,8192 };

    struct CaseConfig {
        DataPattern pX;
        DataPattern pY;
    };

    const CaseConfig configs[] = {
        { COSINE,  COSINE },
        { FILL,    COSINE },
        { COSINE,  FILL   },
        { SPECIAL, COSINE },
        { RANDOM,  RANDOM }
    };

    for (size_t n : sizes) {
        for (const auto & cfg : configs) {
            auto x = generate_data<ggml_fp16_t>(n, cfg.pX, 0.0f);
            auto y = generate_data<ggml_fp16_t>(n, cfg.pY, 0.0f);
            validate_dot_fp16(n, x, y, local_stats);
        }
    }

    printf("Passed: %d/%d\n", local_stats.total_tests - local_stats.failed_tests, local_stats.total_tests);
    global_stats.total_tests += local_stats.total_tests;
    global_stats.failed_tests += local_stats.failed_tests;
}

static void test_vec_dot_bf16_all_cases(TestStats & global_stats) {
    printf("\n--- Testing ggml_vec_dot_bf16 ---\n");
    TestStats    local_stats;
    const size_t sizes[] = { 1, 7, 16, 31, 32, 1024, 1025, 2048,8192 };

    struct CaseConfig {
        DataPattern pX;
        DataPattern pY;
    };

    const CaseConfig configs[] = {
        { COSINE, COSINE  },
        { FILL,   COSINE  },
        { COSINE, FILL    },
        { COSINE, SPECIAL },
        { RANDOM, RANDOM  }
    };

    for (size_t n : sizes) {
        for (const auto & cfg : configs) {
            auto x = generate_data<ggml_bf16_t>(n, cfg.pX, 0.0f);
            auto y = generate_data<ggml_bf16_t>(n, cfg.pY, 0.0f);
            validate_dot_bf16(n, x, y, local_stats);
        }
    }

    printf("Passed: %d/%d\n", local_stats.total_tests - local_stats.failed_tests, local_stats.total_tests);
    global_stats.total_tests += local_stats.total_tests;
    global_stats.failed_tests += local_stats.failed_tests;
}

static void test_cpu_bf16_to_fp32_all_cases(TestStats & global_stats) {
    printf("\n--- Testing ggml_cpu_bf16_to_fp32 ---\n");
    TestStats         local_stats;
    const size_t      sizes[]    = { 1, 7, 16, 31, 32, 1024, 1025, 2048,10000 };

    const DataPattern patterns[] = { COSINE, FILL, SPECIAL };

    for (size_t n : sizes) {
        for (DataPattern p : patterns) {
            auto x = generate_data<ggml_bf16_t>(n, p, 0.0f);
            // Call specific validator
            validate_conversion_bf16(n, x, local_stats);
        }
    }
    printf("Passed: %d/%d\n", local_stats.total_tests - local_stats.failed_tests, local_stats.total_tests);
    global_stats.total_tests += local_stats.total_tests;
    global_stats.failed_tests += local_stats.failed_tests;
}

static void test_cpu_fp16_to_fp32_all_cases(TestStats & global_stats) {
    printf("\n--- Testing ggml_cpu_fp16_to_fp32 ---\n");
    TestStats         local_stats;
    const size_t      sizes[]    = { 1, 7, 16, 31, 32, 1024, 1025, 2048,8192 };
    const DataPattern patterns[] = { COSINE, FILL, SPECIAL };

    for (size_t n : sizes) {
        for (DataPattern p : patterns) {
            auto x = generate_data<ggml_fp16_t>(n, p, 0.0f);
            // Call specific validator
            validate_conversion_fp16(n, x, local_stats);
        }
    }
    printf("Passed: %d/%d\n", local_stats.total_tests - local_stats.failed_tests, local_stats.total_tests);
    global_stats.total_tests += local_stats.total_tests;
    global_stats.failed_tests += local_stats.failed_tests;
}

static void test_vec_silu_f32_all_cases(TestStats & global_stats) {
    printf("\n--- Testing ggml_vec_silu_f32 ---\n");
    TestStats         local_stats;
    const size_t      sizes[]    = { 1, 7, 16, 31, 32, 1024, 1025, 2048,8192 };
    const DataPattern patterns[] = { COSINE, FILL, SPECIAL };

    for (size_t n : sizes) {
        for (DataPattern p : patterns) {
            auto x = generate_data<float>(n, p, 0.0f);
            validate_silu(n, x, local_stats);
        }
    }
    printf("Passed: %d/%d\n", local_stats.total_tests - local_stats.failed_tests, local_stats.total_tests);
    global_stats.total_tests += local_stats.total_tests;
    global_stats.failed_tests += local_stats.failed_tests;
}

static void test_vec_dot_f16_unroll_all_cases(TestStats & global_stats) {
    printf("\n--- Testing ggml_vec_dot_f16_unroll ---\n");
    TestStats         local_stats;
    const size_t      sizes[]    = { 1, 7, 16, 31, 32, 1024, 1025, 2048,8192 };
    const size_t      paddings[] = { 0, 5, 8, 16 };
    const DataPattern patterns[] = { COSINE, FILL };

    for (size_t n : sizes) {
        for (size_t padding : paddings) {
            const int xs = (n + padding) * sizeof(ggml_fp16_t);

            std::vector<char> xv_data(xs * GGML_VEC_DOT_UNROLL);
            void *            xv = xv_data.data();

            for (DataPattern p : patterns) {
                auto y = generate_data<ggml_fp16_t>(n, p, 0.0f);

                // Fill unrolled vectors
                for (int k = 0; k < GGML_VEC_DOT_UNROLL; ++k) {
                    ggml_fp16_t * xv_k = (ggml_fp16_t *) ((char *) xv + k * xs);
                    auto          tmp  = generate_data<ggml_fp16_t>(n, COSINE);
                    memcpy(xv_k, tmp.data(), n * sizeof(ggml_fp16_t));
                }
                validate_dot_unroll(n, xs, xv, y, local_stats);
            }
        }
    }
    printf("Passed: %d/%d\n", local_stats.total_tests - local_stats.failed_tests, local_stats.total_tests);
    global_stats.total_tests += local_stats.total_tests;
    global_stats.failed_tests += local_stats.failed_tests;
}

int main() {
    ggml_cpu_init();

    // Seed with random device for reproducibility of failures
    std::random_device rd;
    unsigned int       seed = rd();
    printf("Executing Kernel Tests.\n");
    printf("Random Seed: %u\n", seed);
    g_rng.seed(seed);

    TestStats global_stats;

    test_vec_scale_f16_all_cases(global_stats);
    test_vec_mad_f16_all_cases(global_stats);
    test_vec_dot_f16_all_cases(global_stats);
    test_vec_dot_bf16_all_cases(global_stats);
    test_vec_dot_f16_unroll_all_cases(global_stats);
    test_cpu_bf16_to_fp32_all_cases(global_stats);
    test_cpu_fp16_to_fp32_all_cases(global_stats);
    test_vec_silu_f32_all_cases(global_stats);

    printf("\n----------------------------------------------------------\n");
    printf("Total Tests: %d\n", global_stats.total_tests);
    printf("Total Failures: %d\n", global_stats.failed_tests);

    return global_stats.failed_tests > 0 ? 1 : 0;
}
