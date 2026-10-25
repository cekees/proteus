// One macro for the kernels' functions, so that the same source is the host prototype's and a Kokkos
// device kernel's. Define ADRH_USE_KOKKOS (and include Kokkos_Core.hpp) before the first adrh header to get
// KOKKOS_INLINE_FUNCTION; otherwise the functions are plain inline and nothing here needs Kokkos.
#pragma once
#ifdef ADRH_USE_KOKKOS
#include <Kokkos_Core.hpp>
#define ADRH_FN KOKKOS_INLINE_FUNCTION
#else
#define ADRH_FN inline
#endif

// The per-element function is called once per element from a loop (host) or a thread (device). Left to its own
// heuristics a host compiler stopped inlining it into the loop, and the small Poisson kernel got 3.5 times slower
// (0.25 to 0.87 microseconds per element, Apple clang 21, -O3); forcing the inline restores the old code.
#if defined(__GNUC__) || defined(__clang__)
#define ADRH_FORCE_INLINE __attribute__((always_inline))
#else
#define ADRH_FORCE_INLINE
#endif
// The compile-time loops (constexpr_for) call a lambda per index. Called from a member function or a Kokkos functor,
// one of them was left as a real call, which forces every array the loop body touches into memory: the small
// Poisson kernel ran 3.2 times slower (815 against 250 ns per element) than the same code in a plain loop.
#define ADRH_LAMBDA_INLINE ADRH_FORCE_INLINE

// Inputs and outputs of the per-element function do not overlap. Said to the compiler, it can keep the quadrature
// data and the element's values in registers across the stores; it cannot know that by itself inside a threaded loop.
#if defined(__GNUC__) || defined(__clang__) || defined(__CUDACC__)
#define ADRH_RESTRICT __restrict__
#else
#define ADRH_RESTRICT
#endif
