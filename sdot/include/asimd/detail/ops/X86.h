#pragma once

// =============================================================================================
// REGISTER-LEVEL VARIANTS FOR x86.
//
// Each one replaces the generic form -- which stays correct, and stays the only one available
// elsewhere -- with one or two instructions. Ranks do the sorting, so an AVX-512 form and an AVX
// one coexist at the same width without the ambiguity plain overloading would produce.
//
// COVERAGE IS THE POINT OF THIS FILE, not the mechanism. A width the table does not mention
// silently gets the generic path -- and the native width on an AVX-512 machine is 16 floats, so
// a table stopping at eight would give `SimdVec<float>`, written with no explicit width, the
// slowest path on the widest hardware (`fma`: 23 instructions with 12 stack accesses, against 3
// and 0). `tests/test_x86_dispatch.cpp` calls `require_at_least` at each width the library
// claims to support, so a missing row fails the build rather than the benchmark.
//
// Laid out by feature level:
//
//   1. 128 bits -- SSE2, SSE4.1, FMA
//   2. 256 bits -- AVX, AVX2, FMA
//   3. 512 bits -- AVX-512F
//   4. mask registers at 128 and 256 bits -- AVX-512VL
//
// A note on the two flavours of mask. Below AVX-512 a comparison yields a LANE mask (all-ones
// words) and `select` is a `blendv`; from AVX-512 on it yields a BIT mask in a `k` register and
// `select` is a masked move. Both are registered, at ranks REGISTER and MASK_REGISTER, and the
// caller never has to name which -- `gt(...)` returns whichever the target has, and `select` and
// `to_bits` accept both. That is what the `IS` slot of `Key` carries.
// =============================================================================================

#include "../impl/x86_intrin.h"

#if defined( _M_X64 ) || defined( _M_IX86 ) || defined( __i386__ ) || defined( __x86_64__ )

#include "../architectures/X86CpuFeatures.h"

// The variant SHAPES -- one macro per (operation, mask flavour) -- are in `Shapes.h`, shared
// with the other backends. This file is the TABLE, which is the part that is actually about x86.
#include "Shapes.h"

namespace asimd {

// =============================================================================================
// 1. 128 BITS -- SSE2, SSE4.1, FMA
// =============================================================================================
#ifdef ASIMD_X86_HAS_SSE2

// ---- comparisons, lane flavour. `cmpps` / `pcmpgtd` are SSE2; there is no predicate operand,
// so each relation is its own instruction, and `ge` on the integer types is `not( a < b )`.
ASIMD_OPS_CMP( SSE2, cmp_gt, FP32, 4, 32, REGISTER, _mm_castps_si128( _mm_cmpgt_ps( a.data.reg, b.data.reg ) ) );
ASIMD_OPS_CMP( SSE2, cmp_lt, FP32, 4, 32, REGISTER, _mm_castps_si128( _mm_cmplt_ps( a.data.reg, b.data.reg ) ) );
ASIMD_OPS_CMP( SSE2, cmp_eq, FP32, 4, 32, REGISTER, _mm_castps_si128( _mm_cmpeq_ps( a.data.reg, b.data.reg ) ) );
ASIMD_OPS_CMP( SSE2, cmp_ge, FP32, 4, 32, REGISTER, _mm_castps_si128( _mm_cmpge_ps( a.data.reg, b.data.reg ) ) );
ASIMD_OPS_CMP( SSE2, cmp_gt, FP64, 2, 64, REGISTER, _mm_castpd_si128( _mm_cmpgt_pd( a.data.reg, b.data.reg ) ) );
ASIMD_OPS_CMP( SSE2, cmp_lt, FP64, 2, 64, REGISTER, _mm_castpd_si128( _mm_cmplt_pd( a.data.reg, b.data.reg ) ) );
ASIMD_OPS_CMP( SSE2, cmp_eq, FP64, 2, 64, REGISTER, _mm_castpd_si128( _mm_cmpeq_pd( a.data.reg, b.data.reg ) ) );
ASIMD_OPS_CMP( SSE2, cmp_ge, FP64, 2, 64, REGISTER, _mm_castpd_si128( _mm_cmpge_pd( a.data.reg, b.data.reg ) ) );

ASIMD_OPS_CMP( SSE2, cmp_eq, SI32, 4, 32, REGISTER, _mm_cmpeq_epi32( a.data.reg, b.data.reg ) );
ASIMD_OPS_CMP( SSE2, cmp_gt, SI32, 4, 32, REGISTER, _mm_cmpgt_epi32( a.data.reg, b.data.reg ) );
ASIMD_OPS_CMP( SSE2, cmp_lt, SI32, 4, 32, REGISTER, _mm_cmpgt_epi32( b.data.reg, a.data.reg ) );
ASIMD_OPS_CMP( SSE2, cmp_ge, SI32, 4, 32, REGISTER, _mm_xor_si128( _mm_cmpgt_epi32( b.data.reg, a.data.reg ), _mm_set1_epi32( -1 ) ) );

// ---- to_bits. `movmskps` is SSE, `movmskpd` SSE2: the lane mask goes straight to an integer.
ASIMD_OPS_TO_BITS( SSE2, 4, 32, REGISTER, PI64( unsigned( _mm_movemask_ps( _mm_castsi128_ps( m.data.reg ) ) ) ) );
ASIMD_OPS_TO_BITS( SSE2, 2, 64, REGISTER, PI64( unsigned( _mm_movemask_pd( _mm_castsi128_pd( m.data.reg ) ) ) ) );

// ---- mask_from_bits, lane flavour. Broadcast the integer, keep one bit per lane, compare it
// back to itself: three instructions where the generic form writes four lanes one at a time.
ASIMD_OPS_MASK_FROM_BITS( SSE2, 4, 32, REGISTER, ( [ & ] {
    const __m128i sel = _mm_setr_epi32( 1, 2, 4, 8 );
    return _mm_cmpeq_epi32( _mm_and_si128( _mm_set1_epi32( int( PI32( b ) ) ), sel ), sel ); }() ) );

#ifdef ASIMD_X86_HAS_SSE4_1
// ---- select. `blendv` is SSE4.1; below it the generic form applies.
ASIMD_OPS_SELECT( SSE4_1, FP32, 4, 32, REGISTER, _mm_blendv_ps( b.data.reg, a.data.reg, _mm_castsi128_ps( m.data.reg ) ) );
ASIMD_OPS_SELECT( SSE4_1, FP64, 2, 64, REGISTER, _mm_blendv_pd( b.data.reg, a.data.reg, _mm_castsi128_pd( m.data.reg ) ) );
ASIMD_OPS_SELECT( SSE4_1, SI32, 4, 32, REGISTER, _mm_castps_si128( _mm_blendv_ps( _mm_castsi128_ps( b.data.reg ), _mm_castsi128_ps( a.data.reg ), _mm_castsi128_ps( m.data.reg ) ) ) );
ASIMD_OPS_SELECT( SSE4_1, PI32, 4, 32, REGISTER, _mm_castps_si128( _mm_blendv_ps( _mm_castsi128_ps( b.data.reg ), _mm_castsi128_ps( a.data.reg ), _mm_castsi128_ps( m.data.reg ) ) ) );
ASIMD_OPS_SELECT( SSE4_1, SI64, 2, 64, REGISTER, _mm_castpd_si128( _mm_blendv_pd( _mm_castsi128_pd( b.data.reg ), _mm_castsi128_pd( a.data.reg ), _mm_castsi128_pd( m.data.reg ) ) ) );
ASIMD_OPS_SELECT( SSE4_1, PI64, 2, 64, REGISTER, _mm_castpd_si128( _mm_blendv_pd( _mm_castsi128_pd( b.data.reg ), _mm_castsi128_pd( a.data.reg ), _mm_castsi128_pd( m.data.reg ) ) ) );
#endif

// ---- bcast_lane. `shufps` with a repeated index; one instruction, and no memory.
ASIMD_OPS_BCAST( SSE2, FP32, 4, _mm_shuffle_ps( v.data.reg, v.data.reg, _MM_SHUFFLE( LANE, LANE, LANE, LANE ) ) );
ASIMD_OPS_BCAST( SSE2, SI32, 4, _mm_shuffle_epi32( v.data.reg, _MM_SHUFFLE( LANE, LANE, LANE, LANE ) ) );
ASIMD_OPS_BCAST( SSE2, PI32, 4, _mm_shuffle_epi32( v.data.reg, _MM_SHUFFLE( LANE, LANE, LANE, LANE ) ) );
ASIMD_OPS_BCAST( SSE2, FP64, 2, _mm_shuffle_pd   ( v.data.reg, v.data.reg, ( LANE << 1 ) | LANE ) );
ASIMD_OPS_BCAST( SSE2, SI64, 2, _mm_shuffle_epi32( v.data.reg, _MM_SHUFFLE( 2*LANE+1, 2*LANE, 2*LANE+1, 2*LANE ) ) );
ASIMD_OPS_BCAST( SSE2, PI64, 2, _mm_shuffle_epi32( v.data.reg, _MM_SHUFFLE( 2*LANE+1, 2*LANE, 2*LANE+1, 2*LANE ) ) );

// ---- rotate_lanes. ANY permutation of four 32-bit lanes is one `shufps` / `pshufd` with an
// immediate, so the whole register and any prefix of it rotate in one instruction with no
// constant, at every feature level. Two 64-bit lanes are the same instruction seen in pairs.
ASIMD_OPS_ROTATE( SSE2, FP32, 4, _mm_shuffle_ps   ( v.data.reg, v.data.reg, rot::imm4( K, n ) ) );
ASIMD_OPS_ROTATE( SSE2, SI32, 4, _mm_shuffle_epi32( v.data.reg, rot::imm4( K, n ) ) );
ASIMD_OPS_ROTATE( SSE2, PI32, 4, _mm_shuffle_epi32( v.data.reg, rot::imm4( K, n ) ) );
ASIMD_OPS_ROTATE( SSE2, FP64, 2, _mm_shuffle_pd   ( v.data.reg, v.data.reg, rot::src( 0, K, n ) | rot::src( 1, K, n ) << 1 ) );
ASIMD_OPS_ROTATE( SSE2, SI64, 2, _mm_shuffle_epi32( v.data.reg, rot::imm4_of_pairs( K, n ) ) );
ASIMD_OPS_ROTATE( SSE2, PI64, 2, _mm_shuffle_epi32( v.data.reg, rot::imm4_of_pairs( K, n ) ) );

// ---- ext_lanes. `palignr` (SSSE3) is exactly this; below it, two byte shifts and an `or`. The
// integer domain for every type: one bypass cycle on the floating-point ones, against three
// instructions saved.
#define ASIMD_X86_ID( x ) ( x )
#define ASIMD_X86_EXT_128( COND, T, N, TO_I, OF_I ) \
    ASIMD_OPS_EXT_EXCL( COND, SSSE3, T, N, OF_I( _mm_or_si128( \
        _mm_srli_si128( TO_I( a.data.reg ), K * int( sizeof( T ) ) ), \
        _mm_slli_si128( TO_I( b.data.reg ), 16 - K * int( sizeof( T ) ) ) ) ) )
ASIMD_X86_EXT_128( SSE2, FP32, 4, _mm_castps_si128, _mm_castsi128_ps );
ASIMD_X86_EXT_128( SSE2, FP64, 2, _mm_castpd_si128, _mm_castsi128_pd );
ASIMD_X86_EXT_128( SSE2, SI32, 4, ASIMD_X86_ID,     ASIMD_X86_ID );
ASIMD_X86_EXT_128( SSE2, PI32, 4, ASIMD_X86_ID,     ASIMD_X86_ID );
ASIMD_X86_EXT_128( SSE2, SI64, 2, ASIMD_X86_ID,     ASIMD_X86_ID );
ASIMD_X86_EXT_128( SSE2, PI64, 2, ASIMD_X86_ID,     ASIMD_X86_ID );
#undef ASIMD_X86_EXT_128

#ifdef ASIMD_X86_HAS_SSSE3
#define ASIMD_X86_EXT_128( T, N, TO_I, OF_I ) \
    ASIMD_OPS_EXT( SSSE3, T, N, OF_I( _mm_alignr_epi8( TO_I( b.data.reg ), TO_I( a.data.reg ), K * int( sizeof( T ) ) ) ) )
ASIMD_X86_EXT_128( FP32, 4, _mm_castps_si128, _mm_castsi128_ps );
ASIMD_X86_EXT_128( FP64, 2, _mm_castpd_si128, _mm_castsi128_pd );
ASIMD_X86_EXT_128( SI32, 4, ASIMD_X86_ID,     ASIMD_X86_ID );
ASIMD_X86_EXT_128( PI32, 4, ASIMD_X86_ID,     ASIMD_X86_ID );
ASIMD_X86_EXT_128( SI64, 2, ASIMD_X86_ID,     ASIMD_X86_ID );
ASIMD_X86_EXT_128( PI64, 2, ASIMD_X86_ID,     ASIMD_X86_ID );
#undef ASIMD_X86_EXT_128
#endif // ASIMD_X86_HAS_SSSE3

#ifdef ASIMD_X86_HAS_SSSE3
// ---- permute at 128 bits WITHOUT AVX. `pshufb` shuffles bytes, so a 32-bit lane index has to be
// turned into four consecutive byte indices: multiply by four, broadcast the low byte of each
// lane across its four bytes, then add 0,1,2,3. Four instructions instead of one `vpermilps`,
// and it is what makes the split form below reachable on a plain SSE machine -- without it,
// `SimdVec<float,8>` on SSE2 had no permutation better than a round trip through memory.
#define ASIMD_OPS_PSHUFB_32( T ) \
    ASIMD_OPS_PERMUTE_EXCL( SSSE3, AVX, T, 4, ( [ & ] { \
        const __m128i bcast = _mm_setr_epi8( 0,0,0,0, 4,4,4,4, 8,8,8,8, 12,12,12,12 ); \
        const __m128i lane  = _mm_setr_epi8( 0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3 ); \
        __m128i c = _mm_shuffle_epi8( _mm_slli_epi32( idx.data.reg, 2 ), bcast ); \
        c = _mm_add_epi8( c, lane ); \
        return ASIMD_PSHUFB_AS( T )( c ); }() ) )

#define ASIMD_PSHUFB_AS_FP32( c ) _mm_castsi128_ps( _mm_shuffle_epi8( _mm_castps_si128( v.data.reg ), c ) )
#define ASIMD_PSHUFB_AS_SI32( c ) _mm_shuffle_epi8( v.data.reg, c )
#define ASIMD_PSHUFB_AS_PI32( c ) _mm_shuffle_epi8( v.data.reg, c )
#define ASIMD_PSHUFB_AS( T ) ASIMD_PSHUFB_AS_##T

ASIMD_OPS_PSHUFB_32( FP32 );
ASIMD_OPS_PSHUFB_32( SI32 );
ASIMD_OPS_PSHUFB_32( PI32 );

#undef ASIMD_PSHUFB_AS
#undef ASIMD_PSHUFB_AS_PI32
#undef ASIMD_PSHUFB_AS_SI32
#undef ASIMD_PSHUFB_AS_FP32
#undef ASIMD_OPS_PSHUFB_32
#endif // ASIMD_X86_HAS_SSSE3

#ifdef ASIMD_X86_HAS_AVX
// ---- permute at 128 bits. `vpermilps` takes its control from a register, which is what makes it
// a VARIABLE-index permutation, and it beats the `pshufb` sequence above by three instructions.
ASIMD_OPS_PERMUTE( AVX, FP32, 4, _mm_permutevar_ps( v.data.reg, idx.data.reg ) );
ASIMD_OPS_PERMUTE( AVX, SI32, 4, _mm_castps_si128( _mm_permutevar_ps( _mm_castsi128_ps( v.data.reg ), idx.data.reg ) ) );
ASIMD_OPS_PERMUTE( AVX, PI32, 4, _mm_castps_si128( _mm_permutevar_ps( _mm_castsi128_ps( v.data.reg ), idx.data.reg ) ) );

// ---- partial load / store: `vmaskmovps` / `vmaskmovpd`, AVX. They read and write ONLY the lanes
// whose mask lane has its top bit set, and fault on none of the others -- which is the contract.
// The mask is a constant for a static set and is built from the bits otherwise: broadcast,
// keep one bit per lane, compare back (SSE2 integer ops, so this works under AVX alone). The
// 64-bit lanes are compared as pairs of 32-bit ones with the same bit in both. Every type goes
// through the `ps` / `pd` form: the instruction moves bits. Gives way to AVX-512VL's masked
// encodings, which take the bits directly.
namespace internal {
    template<LaneSet S,int LANE_BYTES,int N>
    __m128i x86_lane_mask_128( const S &set ) {
        if constexpr ( S::is_static ) {
            return _mm_load_si128( (const __m128i *) LanePattern<S,typename PI_<8 * LANE_BYTES>::T,N>::v.data() );
        } else if constexpr ( LANE_BYTES == 4 ) {
            const __m128i sel = _mm_setr_epi32( 1, 2, 4, 8 );
            return _mm_cmpeq_epi32( _mm_and_si128( _mm_set1_epi32( int( set.bits( 4 ) ) ), sel ), sel );
        } else {
            const __m128i sel = _mm_setr_epi32( 1, 1, 2, 2 );
            return _mm_cmpeq_epi32( _mm_and_si128( _mm_set1_epi32( int( set.bits( 2 ) ) ), sel ), sel );
        }
    }
}
#define ASIMD_X86_PARTIAL_128( T, N, SUF, PTR_T, OF, TO ) \
    ASIMD_OPS_LOAD_PARTIAL ( ASIMD_OPS_REQ_EXCL( AVX, AVX512VL ), T, N, OF( _mm_maskload_##SUF( (const PTR_T *) ptr, internal::x86_lane_mask_128<S,sizeof( T ),N>( set ) ) ) ); \
    ASIMD_OPS_STORE_PARTIAL( ASIMD_OPS_REQ_EXCL( AVX, AVX512VL ), T, N, _mm_maskstore_##SUF( (PTR_T *) ptr, internal::x86_lane_mask_128<S,sizeof( T ),N>( set ), TO( v.data.reg ) ) )
ASIMD_X86_PARTIAL_128( FP32, 4, ps, float,  ASIMD_X86_ID,     ASIMD_X86_ID );
ASIMD_X86_PARTIAL_128( SI32, 4, ps, float,  _mm_castps_si128, _mm_castsi128_ps );
ASIMD_X86_PARTIAL_128( PI32, 4, ps, float,  _mm_castps_si128, _mm_castsi128_ps );
ASIMD_X86_PARTIAL_128( FP64, 2, pd, double, ASIMD_X86_ID,     ASIMD_X86_ID );
ASIMD_X86_PARTIAL_128( SI64, 2, pd, double, _mm_castpd_si128, _mm_castsi128_pd );
ASIMD_X86_PARTIAL_128( PI64, 2, pd, double, _mm_castpd_si128, _mm_castsi128_pd );
#undef ASIMD_X86_PARTIAL_128
#endif

#ifdef ASIMD_X86_HAS_FMA
ASIMD_OPS_FMA( SSE2, FMA, FP32, 4, _mm_fmadd_ps );
ASIMD_OPS_FMA( SSE2, FMA, FP64, 2, _mm_fmadd_pd );
#endif

#endif // ASIMD_X86_HAS_SSE2

// =============================================================================================
// 2. 256 BITS -- AVX, AVX2, FMA
// =============================================================================================
#ifdef ASIMD_X86_HAS_AVX

// ---- comparisons. AVX finally has a predicate operand, but only for floating point: the 256-bit
// integer compares are AVX2, and stay signed-only there.
#define ASIMD_OPS_AVX_FP_CMP( TAG, PRED ) \
    ASIMD_OPS_CMP( AVX, TAG, FP32, 8, 32, REGISTER, _mm256_castps_si256( _mm256_cmp_ps( a.data.reg, b.data.reg, PRED ) ) ); \
    ASIMD_OPS_CMP( AVX, TAG, FP64, 4, 64, REGISTER, _mm256_castpd_si256( _mm256_cmp_pd( a.data.reg, b.data.reg, PRED ) ) )

ASIMD_OPS_AVX_FP_CMP( cmp_gt, _CMP_GT_OQ );
ASIMD_OPS_AVX_FP_CMP( cmp_lt, _CMP_LT_OQ );
ASIMD_OPS_AVX_FP_CMP( cmp_eq, _CMP_EQ_OQ );
ASIMD_OPS_AVX_FP_CMP( cmp_ge, _CMP_GE_OQ );

#undef ASIMD_OPS_AVX_FP_CMP

// ---- to_bits and select at 256 bits. `vblendvps` is AVX -- unlike its 128-bit ancestor, which
// needs SSE4.1 -- so no extra guard here.
ASIMD_OPS_TO_BITS( AVX, 8, 32, REGISTER, PI64( unsigned( _mm256_movemask_ps( _mm256_castsi256_ps( m.data.reg ) ) ) ) );
ASIMD_OPS_TO_BITS( AVX, 4, 64, REGISTER, PI64( unsigned( _mm256_movemask_pd( _mm256_castsi256_pd( m.data.reg ) ) ) ) );

ASIMD_OPS_SELECT( AVX, FP32, 8, 32, REGISTER, _mm256_blendv_ps( b.data.reg, a.data.reg, _mm256_castsi256_ps( m.data.reg ) ) );
ASIMD_OPS_SELECT( AVX, FP64, 4, 64, REGISTER, _mm256_blendv_pd( b.data.reg, a.data.reg, _mm256_castsi256_pd( m.data.reg ) ) );

#define ASIMD_OPS_AVX_ISELECT( T, N, IS, TO, BACK, BLEND ) \
    ASIMD_OPS_SELECT( AVX, T, N, IS, REGISTER, BACK( BLEND( TO( b.data.reg ), TO( a.data.reg ), TO( m.data.reg ) ) ) )
ASIMD_OPS_AVX_ISELECT( SI32, 8, 32, _mm256_castsi256_ps, _mm256_castps_si256, _mm256_blendv_ps );
ASIMD_OPS_AVX_ISELECT( PI32, 8, 32, _mm256_castsi256_ps, _mm256_castps_si256, _mm256_blendv_ps );
ASIMD_OPS_AVX_ISELECT( SI64, 4, 64, _mm256_castsi256_pd, _mm256_castpd_si256, _mm256_blendv_pd );
ASIMD_OPS_AVX_ISELECT( PI64, 4, 64, _mm256_castsi256_pd, _mm256_castpd_si256, _mm256_blendv_pd );
#undef ASIMD_OPS_AVX_ISELECT

#ifdef ASIMD_X86_HAS_FMA
ASIMD_OPS_FMA( AVX, FMA, FP32, 8, _mm256_fmadd_ps );
ASIMD_OPS_FMA( AVX, FMA, FP64, 4, _mm256_fmadd_pd );
#endif

// ---- a FULL 8-lane permutation on AVX, without AVX2 ------------------------------------------
//
// `vpermps` is AVX2. AVX has only `vpermilps`, which permutes inside each 128-bit half. So:
// permute in place, permute again with the halves swapped, and blend by asking whether each
// index points at the half it started in. That question is `( idx ^ lane_half ) & 4`, and the
// shift that moves bit 2 to the sign bit is done per 128-bit half because `vpslld` at 256 bits
// is, once more, AVX2.
//
// Eight instructions against one, and against the eighty-one of the generic form -- which stores
// the vector to the stack, indexes it there and reloads it.
#define ASIMD_OPS_AVX_PERM8( T, TO_PS, FROM_PS ) \
    ASIMD_OPS_PERMUTE_EXCL( AVX, AVX2, T, 8, ( [ & ] { \
        const __m256 s = TO_PS( v.data.reg ); \
        const __m256 in_place = _mm256_permutevar_ps( s, idx.data.reg ); \
        const __m256 crossed  = _mm256_permutevar_ps( _mm256_permute2f128_ps( s, s, 0x01 ), idx.data.reg ); \
        const __m128i j0 = _mm_slli_epi32( _mm256_extractf128_si256( idx.data.reg, 0 ), 29 ); \
        const __m128i j1 = _mm_slli_epi32( _mm_xor_si128( _mm256_extractf128_si256( idx.data.reg, 1 ), \
                                                          _mm_set1_epi32( 4 ) ), 29 ); \
        const __m256 take_crossed = _mm256_castsi256_ps( \
            _mm256_insertf128_si256( _mm256_castsi128_si256( j0 ), j1, 1 ) ); \
        return FROM_PS( _mm256_blendv_ps( in_place, crossed, take_crossed ) ); }() ) )

#define ASIMD_ID_PS( x ) ( x )
ASIMD_OPS_AVX_PERM8( FP32, ASIMD_ID_PS, ASIMD_ID_PS );
ASIMD_OPS_AVX_PERM8( SI32, _mm256_castsi256_ps, _mm256_castps_si256 );
ASIMD_OPS_AVX_PERM8( PI32, _mm256_castsi256_ps, _mm256_castps_si256 );
#undef ASIMD_ID_PS
#undef ASIMD_OPS_AVX_PERM8

// ---- partial load / store at 256 bits. `vmaskmovps` again. The dynamic mask needs 256-bit
// integer compares, which are AVX2: under AVX alone only a static set gets the instruction and a
// dynamic one goes lane by lane.
namespace internal {
    template<LaneSet S,int LANE_BYTES,int N>
    __m256i x86_lane_mask_256( const S &set ) {
        if constexpr ( S::is_static ) {
            return _mm256_load_si256( (const __m256i *) LanePattern<S,typename PI_<8 * LANE_BYTES>::T,N>::v.data() );
        } else if constexpr ( LANE_BYTES == 4 ) {
            const __m256i sel = _mm256_setr_epi32( 1, 2, 4, 8, 16, 32, 64, 128 );
            return _mm256_cmpeq_epi32( _mm256_and_si256( _mm256_set1_epi32( int( set.bits( 8 ) ) ), sel ), sel );
        } else {
            const __m256i sel = _mm256_setr_epi64x( 1, 2, 4, 8 );
            return _mm256_cmpeq_epi64( _mm256_and_si256( _mm256_set1_epi64x( (long long) set.bits( 4 ) ), sel ), sel );
        }
    }
}
#define ASIMD_X86_PARTIAL_256( REQ, T, N, SUF, PTR_T, OF, TO ) \
    ASIMD_OPS_LOAD_PARTIAL ( REQ, T, N, ( [ & ] { \
        if constexpr ( S::is_static || Arch::template Has<features::AVX2>::value ) \
            return OF( _mm256_maskload_##SUF( (const PTR_T *) ptr, internal::x86_lane_mask_256<S,sizeof( T ),N>( set ) ) ); \
        else \
            return sel::Variant<ops::load_partial,Key<T,N,Arch>,sel::GENERIC>::run( ptr, set ).data.reg; }() ) ); \
    ASIMD_OPS_STORE_PARTIAL( REQ, T, N, ( [ & ] { \
        if constexpr ( S::is_static || Arch::template Has<features::AVX2>::value ) \
            _mm256_maskstore_##SUF( (PTR_T *) ptr, internal::x86_lane_mask_256<S,sizeof( T ),N>( set ), TO( v.data.reg ) ); \
        else \
            sel::Variant<ops::store_partial,Key<T,N,Arch>,sel::GENERIC>::run( ptr, v, set ); }() ) )
ASIMD_X86_PARTIAL_256( ASIMD_OPS_REQ_EXCL( AVX, AVX512VL ), FP32, 8, ps, float,  ASIMD_X86_ID,        ASIMD_X86_ID );
ASIMD_X86_PARTIAL_256( ASIMD_OPS_REQ_EXCL( AVX, AVX512VL ), SI32, 8, ps, float,  _mm256_castps_si256, _mm256_castsi256_ps );
ASIMD_X86_PARTIAL_256( ASIMD_OPS_REQ_EXCL( AVX, AVX512VL ), PI32, 8, ps, float,  _mm256_castps_si256, _mm256_castsi256_ps );
ASIMD_X86_PARTIAL_256( ASIMD_OPS_REQ_EXCL( AVX, AVX512VL ), FP64, 4, pd, double, ASIMD_X86_ID,        ASIMD_X86_ID );
ASIMD_X86_PARTIAL_256( ASIMD_OPS_REQ_EXCL( AVX, AVX512VL ), SI64, 4, pd, double, _mm256_castpd_si256, _mm256_castsi256_pd );
ASIMD_X86_PARTIAL_256( ASIMD_OPS_REQ_EXCL( AVX, AVX512VL ), PI64, 4, pd, double, _mm256_castpd_si256, _mm256_castsi256_pd );
#undef ASIMD_X86_PARTIAL_256

// ---- rotate_lanes and ext_lanes at 256 bits WITHOUT AVX2. There is no `vpalignr ymm` and no
// `vpermps` here: what AVX has is `vperm2f128` to swap or pair the two 128-bit halves and
// `vshufps` / `vshufpd`, which pick two lanes from each of two sources within a 128-bit lane.
// A per-lane `x[K..4) ++ y[0..K)` on 32-bit lanes is one `vshufps` for K = 2 and two for
// K = 1 or 3; on 64-bit lanes it is one `vshufpd`. A whole-register rotation is that applied to
// `( v, swapped v )`, a two-source `ext` to `( a, [a.hi, b.lo] )`, and a prefix rotation that
// fits the low lane is an in-lane `vpermilps` and a blend. Without these a `ymm` has no split to
// fall back on and the generic form is a round trip through memory.
namespace internal {
    template<int K> HaD __m256 avx_lane_ext_ps( __m256 x, __m256 y ) {
        if constexpr ( K == 0 ) return x;
        else if constexpr ( K == 2 ) return _mm256_shuffle_ps( x, y, _MM_SHUFFLE( 1, 0, 3, 2 ) );
        else if constexpr ( K == 1 ) {
            const __m256 t = _mm256_shuffle_ps( x, y, _MM_SHUFFLE( 0, 0, 3, 3 ) );   // x3 x3 y0 y0
            return _mm256_shuffle_ps( x, t, _MM_SHUFFLE( 2, 0, 2, 1 ) );             // x1 x2 t0 t2
        } else {
            const __m256 t = _mm256_shuffle_ps( x, y, _MM_SHUFFLE( 0, 0, 3, 3 ) );   // x3 x3 y0 y0
            return _mm256_shuffle_ps( t, y, _MM_SHUFFLE( 2, 1, 2, 0 ) );             // t0 t2 y1 y2
        }
    }
    template<int K> HaD __m256d avx_lane_ext_pd( __m256d x, __m256d y ) {
        if constexpr ( K == 0 ) return x;
        else return _mm256_shuffle_pd( x, y, 0x5 );                                  // x1 y0
    }
}

#define ASIMD_X86_AVX1_EXT( T, N, TO, OF, P2F128, LANE_EXT, L ) \
    ASIMD_OPS_EXT_EXCL( AVX, AVX2, T, N, ( [ & ] { \
        const auto lo = TO( a.data.reg ), hi = TO( b.data.reg ); \
        const auto mid = P2F128( lo, hi, 0x21 ); \
        if constexpr ( K < L ) return OF( internal::LANE_EXT<K>( lo, mid ) ); \
        else if constexpr ( K == L ) return OF( mid ); \
        else return OF( internal::LANE_EXT<K - L>( mid, hi ) ); }() ) ); \
    ASIMD_OPS_ROTATE_IF_EXCL( AVX, AVX2, n == N, T, N, ( [ & ] { \
        const auto x = TO( v.data.reg ); \
        const auto t = P2F128( x, x, 0x01 ); \
        if constexpr ( K < L ) return OF( internal::LANE_EXT<K>( x, t ) ); \
        else if constexpr ( K == L ) return OF( t ); \
        else return OF( internal::LANE_EXT<K - L>( t, x ) ); }() ) )

ASIMD_X86_AVX1_EXT( FP32, 8, ASIMD_X86_ID,        ASIMD_X86_ID,        _mm256_permute2f128_ps, avx_lane_ext_ps, 4 );
ASIMD_X86_AVX1_EXT( SI32, 8, _mm256_castsi256_ps, _mm256_castps_si256, _mm256_permute2f128_ps, avx_lane_ext_ps, 4 );
ASIMD_X86_AVX1_EXT( PI32, 8, _mm256_castsi256_ps, _mm256_castps_si256, _mm256_permute2f128_ps, avx_lane_ext_ps, 4 );
ASIMD_X86_AVX1_EXT( FP64, 4, ASIMD_X86_ID,        ASIMD_X86_ID,        _mm256_permute2f128_pd, avx_lane_ext_pd, 2 );
ASIMD_X86_AVX1_EXT( SI64, 4, _mm256_castsi256_pd, _mm256_castpd_si256, _mm256_permute2f128_pd, avx_lane_ext_pd, 2 );
ASIMD_X86_AVX1_EXT( PI64, 4, _mm256_castsi256_pd, _mm256_castpd_si256, _mm256_permute2f128_pd, avx_lane_ext_pd, 2 );
#undef ASIMD_X86_AVX1_EXT

// a prefix that fits the low 128-bit lane: an in-lane shuffle with an immediate, then a blend
// that keeps the high lane of `v`.
ASIMD_OPS_ROTATE_IF_EXCL( AVX, AVX2, n <= 4, FP32, 8, _mm256_blend_ps( v.data.reg, _mm256_permute_ps( v.data.reg, rot::imm4( K, n ) ), 0x0F ) );
ASIMD_OPS_ROTATE_IF_EXCL( AVX, AVX2, n <= 4, SI32, 8, _mm256_castps_si256( _mm256_blend_ps( _mm256_castsi256_ps( v.data.reg ), _mm256_permute_ps( _mm256_castsi256_ps( v.data.reg ), rot::imm4( K, n ) ), 0x0F ) ) );
ASIMD_OPS_ROTATE_IF_EXCL( AVX, AVX2, n <= 4, PI32, 8, _mm256_castps_si256( _mm256_blend_ps( _mm256_castsi256_ps( v.data.reg ), _mm256_permute_ps( _mm256_castsi256_ps( v.data.reg ), rot::imm4( K, n ) ), 0x0F ) ) );
ASIMD_OPS_ROTATE_IF_EXCL( AVX, AVX2, n <= 2, FP64, 4, _mm256_blend_pd( v.data.reg, _mm256_permute_pd( v.data.reg, rot::src( 0, K, n ) | rot::src( 1, K, n ) << 1 ), 0x3 ) );
ASIMD_OPS_ROTATE_IF_EXCL( AVX, AVX2, n <= 2, SI64, 4, _mm256_castpd_si256( _mm256_blend_pd( _mm256_castsi256_pd( v.data.reg ), _mm256_permute_pd( _mm256_castsi256_pd( v.data.reg ), rot::src( 0, K, n ) | rot::src( 1, K, n ) << 1 ), 0x3 ) ) );
ASIMD_OPS_ROTATE_IF_EXCL( AVX, AVX2, n <= 2, PI64, 4, _mm256_castpd_si256( _mm256_blend_pd( _mm256_castsi256_pd( v.data.reg ), _mm256_permute_pd( _mm256_castsi256_pd( v.data.reg ), rot::src( 0, K, n ) | rot::src( 1, K, n ) << 1 ), 0x3 ) ) );

#ifdef ASIMD_X86_HAS_AVX2
// ---- 256-bit integer comparisons: signed in hardware, unsigned by flipping both sign bits.
ASIMD_OPS_CMP( AVX2, cmp_eq, SI32, 8, 32, REGISTER, _mm256_cmpeq_epi32( a.data.reg, b.data.reg ) );
ASIMD_OPS_CMP( AVX2, cmp_eq, PI32, 8, 32, REGISTER, _mm256_cmpeq_epi32( a.data.reg, b.data.reg ) );
ASIMD_OPS_CMP( AVX2, cmp_gt, SI32, 8, 32, REGISTER, _mm256_cmpgt_epi32( a.data.reg, b.data.reg ) );
ASIMD_OPS_CMP( AVX2, cmp_lt, SI32, 8, 32, REGISTER, _mm256_cmpgt_epi32( b.data.reg, a.data.reg ) );
ASIMD_OPS_CMP( AVX2, cmp_ge, SI32, 8, 32, REGISTER, _mm256_xor_si256( _mm256_cmpgt_epi32( b.data.reg, a.data.reg ), _mm256_set1_epi32( -1 ) ) );
ASIMD_OPS_CMP( AVX2, cmp_eq, SI64, 4, 64, REGISTER, _mm256_cmpeq_epi64( a.data.reg, b.data.reg ) );
ASIMD_OPS_CMP( AVX2, cmp_eq, PI64, 4, 64, REGISTER, _mm256_cmpeq_epi64( a.data.reg, b.data.reg ) );
ASIMD_OPS_CMP( AVX2, cmp_gt, SI64, 4, 64, REGISTER, _mm256_cmpgt_epi64( a.data.reg, b.data.reg ) );
ASIMD_OPS_CMP( AVX2, cmp_lt, SI64, 4, 64, REGISTER, _mm256_cmpgt_epi64( b.data.reg, a.data.reg ) );
ASIMD_OPS_CMP( AVX2, cmp_ge, SI64, 4, 64, REGISTER, _mm256_xor_si256( _mm256_cmpgt_epi64( b.data.reg, a.data.reg ), _mm256_set1_epi32( -1 ) ) );

#define ASIMD_OPS_AVX2_UCMP( TAG, T, N, IS, W, SIGN, A, B ) \
    ASIMD_OPS_CMP( AVX2, TAG, T, N, IS, REGISTER, _mm256_cmpgt_epi##W( \
        _mm256_xor_si256( A.data.reg, SIGN ), _mm256_xor_si256( B.data.reg, SIGN ) ) )
ASIMD_OPS_AVX2_UCMP( cmp_gt, PI32, 8, 32, 32, _mm256_set1_epi32( int( 0x80000000u ) ), a, b );
ASIMD_OPS_AVX2_UCMP( cmp_lt, PI32, 8, 32, 32, _mm256_set1_epi32( int( 0x80000000u ) ), b, a );
ASIMD_OPS_AVX2_UCMP( cmp_gt, PI64, 4, 64, 64, _mm256_set1_epi64x( SI64( 0x8000000000000000ull ) ), a, b );
ASIMD_OPS_AVX2_UCMP( cmp_lt, PI64, 4, 64, 64, _mm256_set1_epi64x( SI64( 0x8000000000000000ull ) ), b, a );
#undef ASIMD_OPS_AVX2_UCMP

// ---- permutation across the WHOLE 256-bit register. `vpermilps` only moves lanes within each
// 128-bit half; `vpermps` crosses, which is what a permutation has to be able to do.
ASIMD_OPS_PERMUTE( AVX2, FP32, 8, _mm256_permutevar8x32_ps( v.data.reg, idx.data.reg ) );
ASIMD_OPS_PERMUTE( AVX2, SI32, 8, _mm256_permutevar8x32_epi32( v.data.reg, idx.data.reg ) );
ASIMD_OPS_PERMUTE( AVX2, PI32, 8, _mm256_permutevar8x32_epi32( v.data.reg, idx.data.reg ) );

// Broadcasting lane 0 has better than a full permutation: one uop against three.
ASIMD_OPS_BCAST( AVX2, FP32, 8, LANE == 0 ? _mm256_broadcastss_ps( _mm256_castps256_ps128( v.data.reg ) )
                                           : _mm256_permutevar8x32_ps( v.data.reg, _mm256_set1_epi32( LANE ) ) );
ASIMD_OPS_BCAST( AVX2, SI32, 8, LANE == 0 ? _mm256_broadcastd_epi32( _mm256_castsi256_si128( v.data.reg ) )
                                           : _mm256_permutevar8x32_epi32( v.data.reg, _mm256_set1_epi32( LANE ) ) );
ASIMD_OPS_BCAST( AVX2, PI32, 8, LANE == 0 ? _mm256_broadcastd_epi32( _mm256_castsi256_si128( v.data.reg ) )
                                           : _mm256_permutevar8x32_epi32( v.data.reg, _mm256_set1_epi32( LANE ) ) );
ASIMD_OPS_BCAST( AVX2, FP64, 4, _mm256_permute4x64_pd   ( v.data.reg, _MM_SHUFFLE( LANE, LANE, LANE, LANE ) ) );
ASIMD_OPS_BCAST( AVX2, SI64, 4, _mm256_permute4x64_epi64( v.data.reg, _MM_SHUFFLE( LANE, LANE, LANE, LANE ) ) );
ASIMD_OPS_BCAST( AVX2, PI64, 4, _mm256_permute4x64_epi64( v.data.reg, _MM_SHUFFLE( LANE, LANE, LANE, LANE ) ) );

// ---- mask_from_bits, 8 lanes, lane flavour.
ASIMD_OPS_MASK_FROM_BITS( AVX2, 8, 32, REGISTER, ( [ & ] { \
    const __m256i sel = _mm256_setr_epi32( 1, 2, 4, 8, 16, 32, 64, 128 ); \
    return _mm256_cmpeq_epi32( _mm256_and_si256( _mm256_set1_epi32( int( PI32( b ) ) ), sel ), sel ); }() ) );

// ---- rotate_lanes at 256 bits. Four 64-bit lanes: `vpermq` / `vpermpd`, an immediate, one
// instruction. Eight 32-bit lanes: `vpermps` with a constant index -- one instruction plus a
// load the compiler hoists out of any loop, where `vperm2f128 + vpalignr` is two instructions
// and only does the whole register.
ASIMD_OPS_ROTATE( AVX2, FP32, 8, _mm256_permutevar8x32_ps   ( v.data.reg, _mm256_load_si256( (const __m256i *) ( rot::Idx<K,n,8>::v.data() ) ) ) );
ASIMD_OPS_ROTATE( AVX2, SI32, 8, _mm256_permutevar8x32_epi32( v.data.reg, _mm256_load_si256( (const __m256i *) ( rot::Idx<K,n,8>::v.data() ) ) ) );
ASIMD_OPS_ROTATE( AVX2, PI32, 8, _mm256_permutevar8x32_epi32( v.data.reg, _mm256_load_si256( (const __m256i *) ( rot::Idx<K,n,8>::v.data() ) ) ) );
ASIMD_OPS_ROTATE( AVX2, FP64, 4, _mm256_permute4x64_pd   ( v.data.reg, rot::imm4( K, n ) ) );
ASIMD_OPS_ROTATE( AVX2, SI64, 4, _mm256_permute4x64_epi64( v.data.reg, rot::imm4( K, n ) ) );
ASIMD_OPS_ROTATE( AVX2, PI64, 4, _mm256_permute4x64_epi64( v.data.reg, rot::imm4( K, n ) ) );

// ---- ext_lanes at 256 bits. `vpalignr` works PER 128-BIT LANE, so the half that crosses the
// middle is brought in first with `vperm2i128` ("high of a, low of b"), and the two are then
// aligned lane by lane: two instructions, whatever the element width. What is left, at the
// SPLIT rank, is `SimdVec<float,16>` on AVX2 -- two of these.
#define ASIMD_X86_EXT_256( T, N, TO_I, OF_I ) \
    ASIMD_OPS_EXT( AVX2, T, N, ( [ & ] { \
        constexpr int nb = K * int( sizeof( T ) ); \
        const __m256i lo = TO_I( a.data.reg ), hi = TO_I( b.data.reg ); \
        const __m256i mid = _mm256_permute2x128_si256( lo, hi, 0x21 ); \
        if constexpr ( nb < 16 ) return OF_I( _mm256_alignr_epi8( mid, lo, nb ) ); \
        else if constexpr ( nb == 16 ) return OF_I( mid ); \
        else return OF_I( _mm256_alignr_epi8( hi, mid, nb - 16 ) ); }() ) )
ASIMD_X86_EXT_256( FP32, 8, _mm256_castps_si256, _mm256_castsi256_ps );
ASIMD_X86_EXT_256( FP64, 4, _mm256_castpd_si256, _mm256_castsi256_pd );
ASIMD_X86_EXT_256( SI32, 8, ASIMD_X86_ID,        ASIMD_X86_ID );
ASIMD_X86_EXT_256( PI32, 8, ASIMD_X86_ID,        ASIMD_X86_ID );
ASIMD_X86_EXT_256( SI64, 4, ASIMD_X86_ID,        ASIMD_X86_ID );
ASIMD_X86_EXT_256( PI64, 4, ASIMD_X86_ID,        ASIMD_X86_ID );
#undef ASIMD_X86_EXT_256
#endif // ASIMD_X86_HAS_AVX2

#endif // ASIMD_X86_HAS_AVX

// =============================================================================================
// 3. 512 BITS AND MASK REGISTERS -- AVX-512F, and AVX-512VL for the 128/256-bit encodings
//
// THIS IS WHERE THE MISSING COVERAGE HURT MOST. On an AVX-512 machine `SimdSize<float>` is 16,
// so `SimdVec<float>` -- the type you get by not naming a width -- landed on a width where this
// file registered nothing at all. Every comparison returned a lane mask built one lane at a time
// and every `select` was a scalar loop.
// =============================================================================================
#ifdef ASIMD_X86_HAS_AVX512F

// ---- comparisons -> mask registers, 512 bits.
#define ASIMD_OPS_AVX512_CMP( TAG, PRED_F, PRED_I ) \
    ASIMD_OPS_CMP( AVX512, TAG, FP32, 16, 1, MASK_REGISTER, _mm512_cmp_ps_mask   ( a.data.reg, b.data.reg, PRED_F ) ); \
    ASIMD_OPS_CMP( AVX512, TAG, FP64,  8, 1, MASK_REGISTER, _mm512_cmp_pd_mask   ( a.data.reg, b.data.reg, PRED_F ) ); \
    ASIMD_OPS_CMP( AVX512, TAG, SI32, 16, 1, MASK_REGISTER, _mm512_cmp_epi32_mask( a.data.reg, b.data.reg, PRED_I ) ); \
    ASIMD_OPS_CMP( AVX512, TAG, PI32, 16, 1, MASK_REGISTER, _mm512_cmp_epu32_mask( a.data.reg, b.data.reg, PRED_I ) ); \
    ASIMD_OPS_CMP( AVX512, TAG, SI64,  8, 1, MASK_REGISTER, _mm512_cmp_epi64_mask( a.data.reg, b.data.reg, PRED_I ) ); \
    ASIMD_OPS_CMP( AVX512, TAG, PI64,  8, 1, MASK_REGISTER, _mm512_cmp_epu64_mask( a.data.reg, b.data.reg, PRED_I ) )

ASIMD_OPS_AVX512_CMP( cmp_gt, _CMP_GT_OQ, _MM_CMPINT_NLE );
ASIMD_OPS_AVX512_CMP( cmp_lt, _CMP_LT_OQ, _MM_CMPINT_LT  );
ASIMD_OPS_AVX512_CMP( cmp_eq, _CMP_EQ_OQ, _MM_CMPINT_EQ  );
ASIMD_OPS_AVX512_CMP( cmp_ge, _CMP_GE_OQ, _MM_CMPINT_NLT );

#undef ASIMD_OPS_AVX512_CMP

// ---- select driven by a mask register: a masked move, not a blend.
ASIMD_OPS_SELECT( AVX512, FP32, 16, 1, MASK_REGISTER, _mm512_mask_blend_ps   ( m.data.reg, b.data.reg, a.data.reg ) );
ASIMD_OPS_SELECT( AVX512, FP64,  8, 1, MASK_REGISTER, _mm512_mask_blend_pd   ( m.data.reg, b.data.reg, a.data.reg ) );
ASIMD_OPS_SELECT( AVX512, SI32, 16, 1, MASK_REGISTER, _mm512_mask_blend_epi32( m.data.reg, b.data.reg, a.data.reg ) );
ASIMD_OPS_SELECT( AVX512, PI32, 16, 1, MASK_REGISTER, _mm512_mask_blend_epi32( m.data.reg, b.data.reg, a.data.reg ) );
ASIMD_OPS_SELECT( AVX512, SI64,  8, 1, MASK_REGISTER, _mm512_mask_blend_epi64( m.data.reg, b.data.reg, a.data.reg ) );
ASIMD_OPS_SELECT( AVX512, PI64,  8, 1, MASK_REGISTER, _mm512_mask_blend_epi64( m.data.reg, b.data.reg, a.data.reg ) );

// ---- to_bits / mask_from_bits: a `k` register IS the integer, so both are a `kmov`.
ASIMD_OPS_TO_BITS( AVX512, 16, 1, MASK_REGISTER, PI64( m.data.reg ) );
ASIMD_OPS_TO_BITS( AVX512,  8, 1, MASK_REGISTER, PI64( m.data.reg ) );
ASIMD_OPS_TO_BITS( AVX512,  4, 1, MASK_REGISTER, PI64( m.data.reg ) & 0xfu );
ASIMD_OPS_TO_BITS( AVX512,  2, 1, MASK_REGISTER, PI64( m.data.reg ) & 0x3u );

ASIMD_OPS_MASK_FROM_BITS( AVX512, 16, 1, MASK_REGISTER, __mmask16( b ) );
ASIMD_OPS_MASK_FROM_BITS( AVX512,  8, 1, MASK_REGISTER, __mmask8 ( b ) );
ASIMD_OPS_MASK_FROM_BITS( AVX512,  4, 1, MASK_REGISTER, __mmask8 ( b & 0xfu ) );
ASIMD_OPS_MASK_FROM_BITS( AVX512,  2, 1, MASK_REGISTER, __mmask8 ( b & 0x3u ) );

// ---- fma, permute, bcast at 512 bits. `vpermps` becomes `vpermt2ps`-class here: one shot over
// sixteen lanes, where the generic form was sixteen loads through memory.
ASIMD_OPS_FMA( AVX512, AVX512, FP32, 16, _mm512_fmadd_ps );
ASIMD_OPS_FMA( AVX512, AVX512, FP64,  8, _mm512_fmadd_pd );

ASIMD_OPS_PERMUTE( AVX512, FP32, 16, _mm512_permutexvar_ps   ( idx.data.reg, v.data.reg ) );
ASIMD_OPS_PERMUTE( AVX512, SI32, 16, _mm512_permutexvar_epi32( idx.data.reg, v.data.reg ) );
ASIMD_OPS_PERMUTE( AVX512, PI32, 16, _mm512_permutexvar_epi32( idx.data.reg, v.data.reg ) );
// the index vector is `SI32 x 8`, i.e. a __m256i: `vpermpd` reads 64-bit indices, so it is
// widened first. One `vpmovsxdq` plus one `vpermpd`, against eight loads through memory.
ASIMD_OPS_PERMUTE( AVX512, FP64,  8, _mm512_permutexvar_pd   ( _mm512_cvtepi32_epi64( idx.data.reg ), v.data.reg ) );
ASIMD_OPS_PERMUTE( AVX512, SI64,  8, _mm512_permutexvar_epi64( _mm512_cvtepi32_epi64( idx.data.reg ), v.data.reg ) );
ASIMD_OPS_PERMUTE( AVX512, PI64,  8, _mm512_permutexvar_epi64( _mm512_cvtepi32_epi64( idx.data.reg ), v.data.reg ) );

ASIMD_OPS_BCAST( AVX512, FP32, 16, _mm512_permutexvar_ps   ( _mm512_set1_epi32( LANE ), v.data.reg ) );
ASIMD_OPS_BCAST( AVX512, SI32, 16, _mm512_permutexvar_epi32( _mm512_set1_epi32( LANE ), v.data.reg ) );
ASIMD_OPS_BCAST( AVX512, FP64,  8, _mm512_permutexvar_pd   ( _mm512_set1_epi64( LANE ), v.data.reg ) );
ASIMD_OPS_BCAST( AVX512, SI64,  8, _mm512_permutexvar_epi64( _mm512_set1_epi64( LANE ), v.data.reg ) );
ASIMD_OPS_BCAST( AVX512, PI32, 16, _mm512_permutexvar_epi32( _mm512_set1_epi32( LANE ), v.data.reg ) );
ASIMD_OPS_BCAST( AVX512, PI64,  8, _mm512_permutexvar_epi64( _mm512_set1_epi64( LANE ), v.data.reg ) );

// ---- rotate_lanes and ext_lanes at 512 bits. `valignd` / `valignq` is the whole-register
// concatenate-and-shift with an immediate -- the x86 `EXT`, arrived with AVX-512. A prefix
// rotation is a `vpermps` / `vpermpd` with a constant index.
#define ASIMD_X86_ALIGN_512( T, N, SUF, TO_I, OF_I ) \
    ASIMD_OPS_EXT( AVX512, T, N, OF_I( _mm512_alignr_##SUF( TO_I( b.data.reg ), TO_I( a.data.reg ), K ) ) ); \
    ASIMD_OPS_ROTATE_IF( AVX512, n == N, T, N, OF_I( _mm512_alignr_##SUF( TO_I( v.data.reg ), TO_I( v.data.reg ), K ) ) )
ASIMD_X86_ALIGN_512( FP32, 16, epi32, _mm512_castps_si512, _mm512_castsi512_ps );
ASIMD_X86_ALIGN_512( FP64,  8, epi64, _mm512_castpd_si512, _mm512_castsi512_pd );
ASIMD_X86_ALIGN_512( SI32, 16, epi32, ASIMD_X86_ID,        ASIMD_X86_ID );
ASIMD_X86_ALIGN_512( PI32, 16, epi32, ASIMD_X86_ID,        ASIMD_X86_ID );
ASIMD_X86_ALIGN_512( SI64,  8, epi64, ASIMD_X86_ID,        ASIMD_X86_ID );
ASIMD_X86_ALIGN_512( PI64,  8, epi64, ASIMD_X86_ID,        ASIMD_X86_ID );
#undef ASIMD_X86_ALIGN_512

ASIMD_OPS_ROTATE_IF( AVX512, n < 16, FP32, 16, _mm512_permutexvar_ps   ( _mm512_load_si512( ( rot::Idx<K,n,16>::v.data() ) ), v.data.reg ) );
ASIMD_OPS_ROTATE_IF( AVX512, n < 16, SI32, 16, _mm512_permutexvar_epi32( _mm512_load_si512( ( rot::Idx<K,n,16>::v.data() ) ), v.data.reg ) );
ASIMD_OPS_ROTATE_IF( AVX512, n < 16, PI32, 16, _mm512_permutexvar_epi32( _mm512_load_si512( ( rot::Idx<K,n,16>::v.data() ) ), v.data.reg ) );
ASIMD_OPS_ROTATE_IF( AVX512, n < 8, FP64,  8, _mm512_permutexvar_pd   ( _mm512_load_si512( ( rot::Idx<K,n,8,std::int64_t>::v.data() ) ), v.data.reg ) );
ASIMD_OPS_ROTATE_IF( AVX512, n < 8, SI64,  8, _mm512_permutexvar_epi64( _mm512_load_si512( ( rot::Idx<K,n,8,std::int64_t>::v.data() ) ), v.data.reg ) );
ASIMD_OPS_ROTATE_IF( AVX512, n < 8, PI64,  8, _mm512_permutexvar_epi64( _mm512_load_si512( ( rot::Idx<K,n,8,std::int64_t>::v.data() ) ), v.data.reg ) );

// ---- partial load / store: the masked encodings, `{k}{z}` on the load and `{k}` on the store.
// The bits of the set ARE the mask register, so both shapes of set cost the same one `kmov`.
#define ASIMD_X86_PARTIAL_K( REQ, T, N, PFX, SUF, MASK ) \
    ASIMD_OPS_LOAD_PARTIAL ( REQ, T, N, PFX##_maskz_loadu_##SUF( MASK( set.bits( N ) ), ptr ) ); \
    ASIMD_OPS_STORE_PARTIAL( REQ, T, N, PFX##_mask_storeu_##SUF( ptr, MASK( set.bits( N ) ), v.data.reg ) )
ASIMD_X86_PARTIAL_K( ASIMD_OPS_REQ1( AVX512 ), FP32, 16, _mm512, ps,    __mmask16 );
ASIMD_X86_PARTIAL_K( ASIMD_OPS_REQ1( AVX512 ), SI32, 16, _mm512, epi32, __mmask16 );
ASIMD_X86_PARTIAL_K( ASIMD_OPS_REQ1( AVX512 ), PI32, 16, _mm512, epi32, __mmask16 );
ASIMD_X86_PARTIAL_K( ASIMD_OPS_REQ1( AVX512 ), FP64,  8, _mm512, pd,    __mmask8  );
ASIMD_X86_PARTIAL_K( ASIMD_OPS_REQ1( AVX512 ), SI64,  8, _mm512, epi64, __mmask8  );
ASIMD_X86_PARTIAL_K( ASIMD_OPS_REQ1( AVX512 ), PI64,  8, _mm512, epi64, __mmask8  );

#ifdef ASIMD_X86_HAS_AVX512BW
ASIMD_X86_PARTIAL_K( ASIMD_OPS_REQ1( AVX512BW ), SI16, 32, _mm512, epi16, __mmask32 );
ASIMD_X86_PARTIAL_K( ASIMD_OPS_REQ1( AVX512BW ), PI16, 32, _mm512, epi16, __mmask32 );
ASIMD_X86_PARTIAL_K( ASIMD_OPS_REQ1( AVX512BW ), SI8 , 64, _mm512, epi8,  __mmask64 );
ASIMD_X86_PARTIAL_K( ASIMD_OPS_REQ1( AVX512BW ), PI8 , 64, _mm512, epi8,  __mmask64 );
#endif // ASIMD_X86_HAS_AVX512BW

#ifdef ASIMD_X86_HAS_AVX512BW
// ---- 16-bit lanes, AVX-512BW: `vpermw` takes a constant index for either shape of rotation,
// and `vpermi2w` reads a two-table index for `ext_lanes`. The 8-bit lanes would need `vpermb`
// (AVX-512VBMI), which asimd does not declare as a feature; they stay generic.
ASIMD_OPS_ROTATE( AVX512BW, SI16, 32, _mm512_permutexvar_epi16( _mm512_load_si512( ( rot::Idx<K,n,32,std::int16_t>::v.data() ) ), v.data.reg ) );
ASIMD_OPS_ROTATE( AVX512BW, PI16, 32, _mm512_permutexvar_epi16( _mm512_load_si512( ( rot::Idx<K,n,32,std::int16_t>::v.data() ) ), v.data.reg ) );
ASIMD_OPS_EXT( AVX512BW, SI16, 32, _mm512_permutex2var_epi16( a.data.reg, _mm512_load_si512( ( rot::ExtIdx<K,32,std::int16_t>::v.data() ) ), b.data.reg ) );
ASIMD_OPS_EXT( AVX512BW, PI16, 32, _mm512_permutex2var_epi16( a.data.reg, _mm512_load_si512( ( rot::ExtIdx<K,32,std::int16_t>::v.data() ) ), b.data.reg ) );
#endif // ASIMD_X86_HAS_AVX512BW

#ifdef ASIMD_X86_HAS_AVX512VL
// ---- THE SAME, AT 128 AND 256 BITS. This is what AVX-512VL is: the AVX-512 encodings, hence the
// mask registers, on the narrower widths. Without these, a comparison at eight lanes on an
// AVX-512 machine still produced a 256-bit LANE mask and `select` still emitted `vblendvps` --
// measured at 3.07 ns against 2.59 for raw intrinsics on a compare-and-select loop.
#define ASIMD_OPS_VL_CMP( TAG, PRED_F, PRED_I ) \
    ASIMD_OPS_CMP( AVX512VL, TAG, FP32, 8, 1, MASK_REGISTER, _mm256_cmp_ps_mask   ( a.data.reg, b.data.reg, PRED_F ) ); \
    ASIMD_OPS_CMP( AVX512VL, TAG, FP64, 4, 1, MASK_REGISTER, _mm256_cmp_pd_mask   ( a.data.reg, b.data.reg, PRED_F ) ); \
    ASIMD_OPS_CMP( AVX512VL, TAG, SI32, 8, 1, MASK_REGISTER, _mm256_cmp_epi32_mask( a.data.reg, b.data.reg, PRED_I ) ); \
    ASIMD_OPS_CMP( AVX512VL, TAG, PI32, 8, 1, MASK_REGISTER, _mm256_cmp_epu32_mask( a.data.reg, b.data.reg, PRED_I ) ); \
    ASIMD_OPS_CMP( AVX512VL, TAG, SI64, 4, 1, MASK_REGISTER, _mm256_cmp_epi64_mask( a.data.reg, b.data.reg, PRED_I ) ); \
    ASIMD_OPS_CMP( AVX512VL, TAG, PI64, 4, 1, MASK_REGISTER, _mm256_cmp_epu64_mask( a.data.reg, b.data.reg, PRED_I ) ); \
    ASIMD_OPS_CMP( AVX512VL, TAG, FP32, 4, 1, MASK_REGISTER, _mm_cmp_ps_mask      ( a.data.reg, b.data.reg, PRED_F ) ); \
    ASIMD_OPS_CMP( AVX512VL, TAG, FP64, 2, 1, MASK_REGISTER, _mm_cmp_pd_mask      ( a.data.reg, b.data.reg, PRED_F ) ); \
    ASIMD_OPS_CMP( AVX512VL, TAG, SI32, 4, 1, MASK_REGISTER, _mm_cmp_epi32_mask   ( a.data.reg, b.data.reg, PRED_I ) ); \
    ASIMD_OPS_CMP( AVX512VL, TAG, PI32, 4, 1, MASK_REGISTER, _mm_cmp_epu32_mask   ( a.data.reg, b.data.reg, PRED_I ) ); \
    ASIMD_OPS_CMP( AVX512VL, TAG, SI64, 2, 1, MASK_REGISTER, _mm_cmp_epi64_mask   ( a.data.reg, b.data.reg, PRED_I ) ); \
    ASIMD_OPS_CMP( AVX512VL, TAG, PI64, 2, 1, MASK_REGISTER, _mm_cmp_epu64_mask   ( a.data.reg, b.data.reg, PRED_I ) )

ASIMD_OPS_VL_CMP( cmp_gt, _CMP_GT_OQ, _MM_CMPINT_NLE );
ASIMD_OPS_VL_CMP( cmp_lt, _CMP_LT_OQ, _MM_CMPINT_LT  );
ASIMD_OPS_VL_CMP( cmp_eq, _CMP_EQ_OQ, _MM_CMPINT_EQ  );
ASIMD_OPS_VL_CMP( cmp_ge, _CMP_GE_OQ, _MM_CMPINT_NLT );

#undef ASIMD_OPS_VL_CMP

#define ASIMD_OPS_VL_SELECT( T, N, FUNC ) \
    ASIMD_OPS_SELECT( AVX512VL, T, N, 1, MASK_REGISTER, FUNC( m.data.reg, b.data.reg, a.data.reg ) )
ASIMD_OPS_VL_SELECT( FP32, 8, _mm256_mask_blend_ps    );
ASIMD_OPS_VL_SELECT( FP64, 4, _mm256_mask_blend_pd    );
ASIMD_OPS_VL_SELECT( SI32, 8, _mm256_mask_blend_epi32 );
ASIMD_OPS_VL_SELECT( PI32, 8, _mm256_mask_blend_epi32 );
ASIMD_OPS_VL_SELECT( SI64, 4, _mm256_mask_blend_epi64 );
ASIMD_OPS_VL_SELECT( PI64, 4, _mm256_mask_blend_epi64 );
ASIMD_OPS_VL_SELECT( FP32, 4, _mm_mask_blend_ps       );
ASIMD_OPS_VL_SELECT( FP64, 2, _mm_mask_blend_pd       );
ASIMD_OPS_VL_SELECT( SI32, 4, _mm_mask_blend_epi32    );
ASIMD_OPS_VL_SELECT( PI32, 4, _mm_mask_blend_epi32    );
ASIMD_OPS_VL_SELECT( SI64, 2, _mm_mask_blend_epi64    );
ASIMD_OPS_VL_SELECT( PI64, 2, _mm_mask_blend_epi64    );
#undef ASIMD_OPS_VL_SELECT

// a full 4-lane permutation of 64-bit elements needs `vpermpd` with a REGISTER control, which is
// AVX-512VL: the AVX2 `_mm256_permutevar_pd` only moves within each 128-bit half.
ASIMD_OPS_PERMUTE( AVX512VL, FP64, 4, _mm256_permutexvar_pd   ( _mm256_cvtepi32_epi64( idx.data.reg ), v.data.reg ) );
ASIMD_OPS_PERMUTE( AVX512VL, SI64, 4, _mm256_permutexvar_epi64( _mm256_cvtepi32_epi64( idx.data.reg ), v.data.reg ) );

// ---- partial load / store at 128 and 256 bits, the masked encodings again.
ASIMD_X86_PARTIAL_K( ASIMD_OPS_REQ1( AVX512VL ), FP32, 8, _mm256, ps,    __mmask8 );
ASIMD_X86_PARTIAL_K( ASIMD_OPS_REQ1( AVX512VL ), SI32, 8, _mm256, epi32, __mmask8 );
ASIMD_X86_PARTIAL_K( ASIMD_OPS_REQ1( AVX512VL ), PI32, 8, _mm256, epi32, __mmask8 );
ASIMD_X86_PARTIAL_K( ASIMD_OPS_REQ1( AVX512VL ), FP64, 4, _mm256, pd,    __mmask8 );
ASIMD_X86_PARTIAL_K( ASIMD_OPS_REQ1( AVX512VL ), SI64, 4, _mm256, epi64, __mmask8 );
ASIMD_X86_PARTIAL_K( ASIMD_OPS_REQ1( AVX512VL ), PI64, 4, _mm256, epi64, __mmask8 );
ASIMD_X86_PARTIAL_K( ASIMD_OPS_REQ1( AVX512VL ), FP32, 4, _mm,    ps,    __mmask8 );
ASIMD_X86_PARTIAL_K( ASIMD_OPS_REQ1( AVX512VL ), SI32, 4, _mm,    epi32, __mmask8 );
ASIMD_X86_PARTIAL_K( ASIMD_OPS_REQ1( AVX512VL ), PI32, 4, _mm,    epi32, __mmask8 );
ASIMD_X86_PARTIAL_K( ASIMD_OPS_REQ1( AVX512VL ), FP64, 2, _mm,    pd,    __mmask8 );
ASIMD_X86_PARTIAL_K( ASIMD_OPS_REQ1( AVX512VL ), SI64, 2, _mm,    epi64, __mmask8 );
ASIMD_X86_PARTIAL_K( ASIMD_OPS_REQ1( AVX512VL ), PI64, 2, _mm,    epi64, __mmask8 );
#undef ASIMD_X86_PARTIAL_K
#endif // ASIMD_X86_HAS_AVX512VL

#endif // ASIMD_X86_HAS_AVX512F

#undef ASIMD_X86_ID

} // namespace asimd

#endif // x86
