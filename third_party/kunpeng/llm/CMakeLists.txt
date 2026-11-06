# libkunpeng_llm.so: hand-written LLM kernels for Kunpeng CPUs.  The
# triton-shared TritonToKunpengLLM pass replaces recognised Triton kernels by
# calls into it (API: kunpeng_llm.h).  Like libarmpl_lp64.so the library goes
# next to libtriton.so (triton/_C), where triton-shared/backend/
# {compiler,driver}.py look for it.
#
# Needs a C compiler with SME support; skipped otherwise.

include(CheckCSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-march=armv9-a+sme+sve2+bf16")
# The SME ABI support routines (__arm_sme_state, __arm_tpidr2_save, ...) come
# from compiler-rt.
set(CMAKE_REQUIRED_LINK_OPTIONS "-rtlib=compiler-rt")
check_c_source_compiles("
  #include <arm_sme.h>
  __arm_locally_streaming __arm_new(\"za\") void f(void) { svzero_za(); }
  int main(void) { f(); return 0; }" KUNPENG_LLM_HAVE_SME_COMPILER)
unset(CMAKE_REQUIRED_FLAGS)
unset(CMAKE_REQUIRED_LINK_OPTIONS)

if (NOT KUNPENG_LLM_HAVE_SME_COMPILER)
  message(STATUS "C compiler lacks SME support; not building libkunpeng_llm")
  return()
endif()

add_library(kunpeng_llm SHARED flash_attn_fwd_sme.c)
target_compile_options(kunpeng_llm PRIVATE -O3 -march=armv9-a+sme+sve2+bf16)
target_link_options(kunpeng_llm PRIVATE -rtlib=compiler-rt)
target_link_libraries(kunpeng_llm PRIVATE m)
set_target_properties(kunpeng_llm PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
  C_VISIBILITY_PRESET default)
