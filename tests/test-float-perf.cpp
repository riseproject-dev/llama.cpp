#include "ggml-cpu.h"
#include "ggml.h"
#include "vec.h"  // needed because defined kernel implementations are static

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <vector>
#define MAX_ALIGNMENT 64
#define WARMUP        10
#define ITERATIONS    200

static int64_t time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t) ts.tv_sec * 1000000000 + (int64_t) ts.tv_nsec;
}

template <typename T> static void generate_vector(T * data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        float f_val = 2.0f * std::cos(i * 0.1f) - 0.1f;
        if constexpr (std::is_same_v<T, float>) {
            data[i] = f_val;
        } else if constexpr (std::is_same_v<T, ggml_fp16_t>) {
            data[i] = ggml_fp32_to_fp16(f_val);
        } else if constexpr (std::is_same_v<T, ggml_bf16_t>) {
            data[i] = ggml_fp32_to_bf16(f_val);
        } else {
            static_assert(!sizeof(T *), "Unsupported type for generate_vector");
        }
    }
}

//Warmup is performed to make the caches hot. Best and Average Ops throughput of the kernel is calculated.
template <typename Func>
static void benchmark_kernel(size_t                             size,
                             int64_t                            iterations,
                             int64_t                            ops_per_call,  // theoretical ops for the kernel
                             Func&&  func) {
    int64_t total_time_us     = 0;
    int64_t curr_time_us      = 0;
    int64_t min_time_us       = INT64_MAX;

    for (int i = 0; i < WARMUP; i++) {
        func();
    }
    for (int i = 0; i < iterations; i++) {
        const int64_t start_time = time_ns();
        func();
        const int64_t end_time = time_ns();
        curr_time_us           = end_time - start_time;
        total_time_us += curr_time_us;
        if (curr_time_us < min_time_us) {
            min_time_us = curr_time_us;
        }
    }

    double total_ops     = (double) ops_per_call * iterations;
    double avg_time_s    = total_time_us / 1e9;
    double m_ops_per_s   = total_ops / avg_time_s / 1e6;
    double min_time_s    = min_time_us / 1e9;
    double min_ops_per_s = (double) ops_per_call / min_time_s / 1e6;

    printf("| %8zu | %12.4f | %12.4f |\n", size, m_ops_per_s, min_ops_per_s);
}

int main() {
    const char * table_header =
        "|   Size   |  Avg M-Ops/s | Best M-Ops/s |\n"
        "|:--------:|-------------:|-------------:|\n";
    {
        printf("\n### Kernel: ggml_vec_dot_f16\n");
        printf("%s", table_header);

        const std::vector<size_t> test_sizes = { 64, 256, 2048, 4096 };
        float                     result;
        for (size_t size : test_sizes) {
            int64_t       ops   = 2 * size;
            ggml_fp16_t * x_f16 = (ggml_fp16_t *) std::aligned_alloc(MAX_ALIGNMENT, size * sizeof(ggml_fp16_t));
            ggml_fp16_t * y_f16 = (ggml_fp16_t *) std::aligned_alloc(MAX_ALIGNMENT, size * sizeof(ggml_fp16_t));
            generate_vector<ggml_fp16_t>(x_f16, size);
            generate_vector<ggml_fp16_t>(y_f16, size);
            auto opt_fn = [&]() {
                ggml_vec_dot_f16(size, &result, 0, x_f16, 0, y_f16, 0, 1);
                return result;
            };
            benchmark_kernel(size, ITERATIONS, ops, opt_fn);
            free(x_f16);
            free(y_f16);
        }

    }

    {
        printf("\n### Kernel: ggml_vec_dot_f16_unroll (unroll=%d)\n", GGML_VEC_DOT_UNROLL);
        printf("%s", table_header);

        float                     result[GGML_VEC_DOT_UNROLL];
        const std::vector<size_t> test_sizes = { 64, 256, 2048 };
        for (size_t size : test_sizes) {
            int64_t       ops = (int64_t) GGML_VEC_DOT_UNROLL * 2 * size;
            const int     xs  = size * sizeof(ggml_fp16_t);
            ggml_fp16_t * xv_f16_unroll =
                (ggml_fp16_t *) std::aligned_alloc(MAX_ALIGNMENT, GGML_VEC_DOT_UNROLL * size * sizeof(ggml_fp16_t));
            ggml_fp16_t * y_f16 = (ggml_fp16_t *) std::aligned_alloc(MAX_ALIGNMENT, size * sizeof(ggml_fp16_t));
            generate_vector<ggml_fp16_t>(xv_f16_unroll, GGML_VEC_DOT_UNROLL * size);
            generate_vector<ggml_fp16_t>(y_f16, size);
            auto opt_fn = [&]() {
                ggml_vec_dot_f16_unroll(size, xs, result, xv_f16_unroll, y_f16);
                return result[0];
            };
            benchmark_kernel(size, ITERATIONS, ops, opt_fn);
            std::free(xv_f16_unroll);
            std::free(y_f16);
        }
    }

    {
        printf("\n### Kernel: ggml_vec_scale_f16\n");
        printf("%s", table_header);

        const float               v          = 0.5f;
        const std::vector<size_t> test_sizes = {
            64,
            256,
            2048,
        };
        for (size_t size : test_sizes) {
            ggml_fp16_t * y_f16 = (ggml_fp16_t *) std::aligned_alloc(MAX_ALIGNMENT, size * sizeof(ggml_fp16_t));
            generate_vector<ggml_fp16_t>(y_f16, size);
            int64_t ops    = size;
            auto    opt_fn = [&]() {
                ggml_vec_scale_f16(size, y_f16, v);
                return y_f16[0];
            };
            benchmark_kernel(size, ITERATIONS, ops, opt_fn);
            std::free(y_f16);
        }
    }

    {
        printf("\n### Kernel: ggml_vec_mad_f16\n");
        printf("%s", table_header);

        const std::vector<size_t> test_sizes = { 64, 256, 2048 };
        const float               v          = 3.14159f;
        for (size_t size : test_sizes) {
            int64_t       ops   = 2 * size;
            ggml_fp16_t * x_f16 = (ggml_fp16_t *) std::aligned_alloc(MAX_ALIGNMENT, size * sizeof(ggml_fp16_t));
            ggml_fp16_t * y_f16 = (ggml_fp16_t *) std::aligned_alloc(MAX_ALIGNMENT, size * sizeof(ggml_fp16_t));
            generate_vector<ggml_fp16_t>(x_f16, size);
            generate_vector<ggml_fp16_t>(y_f16, size);
            auto opt_fn = [&]() {
                ggml_vec_mad_f16(size, y_f16, x_f16, v);
                return y_f16[0];
            };
            benchmark_kernel(size, ITERATIONS, ops, opt_fn);
            std::free(x_f16);
            std::free(y_f16);
        }
    }

    {
        printf("\n### Kernel: ggml_cpu_bf16_to_fp32\n");
        printf("%s", table_header);

        const std::vector<size_t> test_sizes = { 64, 256, 2048 };

        for (size_t size : test_sizes) {
            int64_t       ops    = size;
            ggml_bf16_t * x_bf16 = (ggml_bf16_t *) std::aligned_alloc(MAX_ALIGNMENT, size * sizeof(ggml_bf16_t));
            float *       y_f32  = (float *) std::aligned_alloc(MAX_ALIGNMENT, size * sizeof(float));
            generate_vector<ggml_bf16_t>(x_bf16, size);
            auto opt_fn = [&]() {
                ggml_cpu_bf16_to_fp32(x_bf16, y_f32, size);
                return y_f32[0];
            };
            benchmark_kernel(size, ITERATIONS, ops, opt_fn);
            std::free(x_bf16);
            std::free(y_f32);
        }
    }

    {
        printf("\n### Kernel: ggml_cpu_fp16_to_fp32\n");
        printf("%s", table_header);

        const std::vector<size_t> test_sizes = { 64, 256, 2048 };

        for (size_t size : test_sizes) {
            int64_t       ops   = size;
            ggml_fp16_t * x_f16 = (ggml_fp16_t *) std::aligned_alloc(MAX_ALIGNMENT, size * sizeof(ggml_fp16_t));
            float *       y_f32 = (float *) std::aligned_alloc(MAX_ALIGNMENT, size * sizeof(float));
            generate_vector<ggml_fp16_t>(x_f16, size);
            auto opt_fn = [&]() {
                ggml_cpu_fp16_to_fp32(x_f16, y_f32, size);
                return y_f32[0];
            };
            benchmark_kernel(size, ITERATIONS, ops, opt_fn);
            std::free(x_f16);
            std::free(y_f32);
        }
    }

    {
        printf("\n### Kernel: ggml_vec_silu_f32\n");
        printf("%s", table_header);

        const std::vector<size_t> test_sizes = { 64, 256, 2048 };
        for (size_t size : test_sizes) {
            int64_t ops   = 33 * size;
            float * y_f32 = (float *) std::aligned_alloc(MAX_ALIGNMENT, size * sizeof(float));
            float * x_f32 = (float *) std::aligned_alloc(MAX_ALIGNMENT, size * sizeof(float));
            generate_vector<float>(x_f32, size);
            auto opt_fn = [&]() {
                ggml_vec_silu_f32(size, y_f32, x_f32);
                return y_f32[0];
            };
            benchmark_kernel(size, ITERATIONS, ops, opt_fn);
            std::free(x_f32);
            std::free(y_f32);
        }
    }

    return 0;
}
