/**
 * @file triad_core.h
 * @brief Unified C-ABI Header for sneed-and-feed/triad-core.
 *
 * Exposes all 6 core mathematical operators alongside their stealth aliases,
 * static arena geometry, and deterministic algorithms for AVX2, NEON, WASM32,
 * and scalar SIMD platforms.
 *
 * License: CC0-1.0 (Public Domain)
 */

#ifndef TRIAD_CORE_H
#define TRIAD_CORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Alignment macro for C and C++ */
#if defined(_MSC_VER)
  #define TRIAD_ALIGN64 __declspec(align(64))
#elif defined(__GNUC__) || defined(__clang__)
  #define TRIAD_ALIGN64 __attribute__((aligned(64)))
#else
  #define TRIAD_ALIGN64
#endif

/* Geometry Constants */
#define TRIAD_SPATIAL_DIM 128
#define TRIAD_MAX_PIXELS (128 * 128)
#define TRIAD_RING_BUFFER_SIZE (TRIAD_MAX_PIXELS * 8)
#define TRIAD_TENSOR_DIM 16
#define TRIAD_TENSOR_SIZE (16 * 16 * 16)
#define TRIAD_CASCADE_DIM 16
#define TRIAD_CASCADE_W_SIZE (16 * 16 * 16)
#define TRIAD_ROTOR_DIM 16
#define TRIAD_MAX_CONTACT_NODES 64
#define TRIAD_MAX_CONTACT_EDGES 128

/* Discrete Morse Classification Codes */
#define TRIAD_MORSE_NOMINAL 0
#define TRIAD_MORSE_PENDING_IMPULSE 1
#define TRIAD_MORSE_STEP_MANEUVER 2
#define TRIAD_MORSE_GLINT_REJECTED 3

/* Contact Node definition (48 bytes) */
typedef struct {
    uint32_t id;
    uint32_t color;
    float x;
    float y;
    float theta;
    float vx;
    float vy;
    float omega;
    float mass;
    float radius;
    uint32_t is_fixed;
    int32_t container_id;
} TriadContactNode;

/* Contact Edge definition (32 bytes) */
typedef struct {
    uint32_t u;
    uint32_t v;
    float normal_x;
    float normal_y;
    float contact_distance;
    uint32_t is_active;
    uint32_t edge_type; /* 0 = unilateral, 1 = container nesting, 2 = peg docking */
    uint32_t _pad;
} TriadContactEdge;

/* 3-Point Discrete Morse Gate State (64 bytes) */
typedef struct {
    float v_curr;
    float v_prev1;
    float v_prev2;
    float pending_v;
    float variance;
    float chi2_threshold;
    uint64_t tau_ref_ns;
    uint64_t pending_dt_ns;
    float pending_nis;
    int32_t gate_status;
    uint32_t total_glints;
    uint32_t total_steps;
    uint32_t initialized;
    uint8_t _pad[4];
} TriadMorseState;

/* Real-Time Telemetry Metrics (64 bytes) */
typedef struct {
    float cable_spike_recovery_rate;
    float glare_rejection_rate;
    float max_norm_drift;
    float max_trace_residual;
    uint32_t total_contact_nodes;
    uint32_t total_contact_edges;
    uint32_t execution_cycles;
    float parity_residual_f32;
    uint8_t _pad[32];
} TriadTelemetry;

/**
 * Universal Static Arena (< 1.2 MB fixed SRAM, 0 heap allocations).
 * Pairwise byte-disjoint sub-buffers, all 64-byte aligned (Theorem 2.11).
 * Total size: exactly 894,912 bytes.
 */
typedef struct TRIAD_ALIGN64 {
    float spatial_input[TRIAD_MAX_PIXELS];              /* 65,536 B */
    float spatial_output[TRIAD_MAX_PIXELS];             /* 65,536 B */
    float salience_map[TRIAD_MAX_PIXELS];               /* 65,536 B */
    float curl_buffer[TRIAD_MAX_PIXELS];                /* 65,536 B */
    float ring_buffer[TRIAD_RING_BUFFER_SIZE];          /* 524,288 B */
    uint8_t mask_clock[TRIAD_MAX_PIXELS];               /* 16,384 B */
    uint8_t mask_nope[TRIAD_MAX_PIXELS];                /* 16,384 B */
    uint8_t mask_glare[TRIAD_MAX_PIXELS];               /* 16,384 B */
    uint8_t mask_deferral[TRIAD_MAX_PIXELS];            /* 16,384 B */
    float triad_tensor_w[TRIAD_TENSOR_SIZE];            /* 16,384 B */
    float cascade_w_rec[TRIAD_CASCADE_W_SIZE];          /* 16,384 B */
    float rotor_so16_a[TRIAD_ROTOR_DIM * TRIAD_ROTOR_DIM]; /* 1,024 B */
    float rotor_so16_r[TRIAD_ROTOR_DIM * TRIAD_ROTOR_DIM]; /* 1,024 B */
    float rotor_v_in[TRIAD_ROTOR_DIM];                  /* 64 B */
    float rotor_v_out[TRIAD_ROTOR_DIM];                 /* 64 B */
    float stiffness_so3_r[16];                          /* 64 B */
    float stiffness_so6_m[48];                          /* 192 B */
    float stiffness_c_base[48];                         /* 192 B */
    float stiffness_c_rot[48];                          /* 192 B */
    TriadContactNode contact_nodes[TRIAD_MAX_CONTACT_NODES]; /* 3,072 B */
    TriadContactEdge contact_edges[TRIAD_MAX_CONTACT_EDGES]; /* 4,096 B */
    TriadMorseState morse_state;                        /* 64 B */
    TriadTelemetry telemetry;                           /* 64 B */
    uint32_t step_counter;
    uint32_t num_nodes;
    uint32_t num_edges;
    uint32_t ring_head;
    uint32_t width;
    uint32_t height;
    float last_radial_drift;
    float last_trace_residual;
    uint8_t _pad[32];
} TriadUniversalArena;

/* ========================================================================= */
/* Operator 1: Skew Triad Operator / so(C) Tie-Breaker                       */
/* ========================================================================= */

int32_t skew_triad_step_f32(
    const float *u,
    const float *v,
    const float *w,
    int32_t dim,
    float gamma,
    float *v_out
);

int32_t so_skew_tiebreaker_f32(
    const float *u,
    const float *v,
    const float *w,
    int32_t dim,
    float gamma,
    float *v_out
);

/* ========================================================================= */
/* Operator 2: Slow-Fast Spherical Cascade Integrator (16-step unroll)      */
/* ========================================================================= */

int32_t slow_fast_cascade_k16_f32(
    const float *u,
    const float *v_in,
    const float *w_rec,
    int32_t k_steps,
    float gamma,
    float *v_out,
    float *radial_drift_out
);

int32_t spherical_cascade_integrator_f32(
    const float *u,
    const float *v_in,
    const float *w_rec,
    int32_t k_steps,
    float gamma,
    float *v_out,
    float *radial_drift_out
);

/* ========================================================================= */
/* Operator 3: Online Rank-2 so(16) Cayley Rotor / Attitude Update           */
/* ========================================================================= */

int32_t cayley_rotor_so16_step_f32(
    uint32_t u_idx,
    uint32_t v_idx,
    float alpha,
    const float *v_in,
    float *v_out,
    float *r_out,
    float *norm_drift_out
);

int32_t cayley_attitude_update_f32(
    uint32_t u_idx,
    uint32_t v_idx,
    float alpha,
    const float *v_in,
    float *v_out,
    float *r_out,
    float *norm_drift_out
);

/* ========================================================================= */
/* Operator 4: Beltrami Exterior Curl + Discrete Morse Gate + MAD Floor      */
/* ========================================================================= */

float compute_spatial_mad_floor_256_c(
    const float *salience,
    const uint8_t *clock_mask,
    size_t total_pixels
);

int32_t beltrami_morse_gate_f32(
    const float *input_frame,
    const float *prev_frame,
    const uint8_t *clock_mask,
    uint32_t width,
    uint32_t height,
    float *salience_out,
    uint8_t *spike_mask_out,
    float *mad_floor_out
);

int32_t morse_innovation_gate_f32(
    const float *input_frame,
    const float *prev_frame,
    const uint8_t *clock_mask,
    uint32_t width,
    uint32_t height,
    float *salience_out,
    uint8_t *spike_mask_out,
    float *mad_floor_out
);

int32_t morse_gate_step_1d_f32(
    TriadMorseState *gate,
    float observation,
    uint64_t delta_t_ns
);

/* ========================================================================= */
/* Operator 5: SO(6) Kelvin-Mandel Stiffness Rotor                           */
/* ========================================================================= */

int32_t kelvin_mandel_so6_rotor_f32(
    const float *r_so3,
    const float *c_base,
    float *m_so6_out,
    float *c_rot_out,
    float *trace_residual_out
);

int32_t anisotropic_stiffness_rotor_f32(
    const float *r_so3,
    const float *c_base,
    float *m_so6_out,
    float *c_rot_out,
    float *trace_residual_out
);

int32_t kelvin_mandel_so6_rotor_f64(
    const double *r_so3,
    const double *c_base,
    double *m_so6_out,
    double *c_rot_out,
    double *trace_residual_out
);

int32_t anisotropic_stiffness_rotor_f64(
    const double *r_so3,
    const double *c_base,
    double *m_so6_out,
    double *c_rot_out,
    double *trace_residual_out
);

/* ========================================================================= */
/* Operator 6: SE(2) Cellular Sheaf Contact Graph & Clutter Filter           */
/* ========================================================================= */

int32_t contact_sheaf_se2_step_f32(
    TriadContactNode *nodes,
    uint32_t num_nodes,
    const TriadContactEdge *edges,
    uint32_t num_edges,
    uint32_t ego_id,
    float dx,
    float dy,
    float *out_dx,
    float *out_dy
);

int32_t rigid_contact_coboundary_solver_f32(
    TriadContactNode *nodes,
    uint32_t num_nodes,
    const TriadContactEdge *edges,
    uint32_t num_edges,
    uint32_t ego_id,
    float dx,
    float dy,
    float *out_dx,
    float *out_dy
);

int32_t continuity_clutter_filter_f32(
    const float *current,
    const float *previous,
    uint8_t *clock_mask,
    uint8_t *nope_mask,
    size_t count,
    float threshold
);

/* ========================================================================= */
/* Arena Lifecycle & Accessors                                              */
/* ========================================================================= */

size_t triad_arena_size_bytes(void);
void triad_arena_init(TriadUniversalArena *arena);
void triad_arena_reset(TriadUniversalArena *arena);
void triad_arena_step(TriadUniversalArena *arena);
TriadUniversalArena *triad_arena_global(void);
TriadUniversalArena *triad_universal_arena_global(void);

#ifdef __cplusplus
}
#endif

#endif /* TRIAD_CORE_H */
