option(TRITON_SHARED_BUILD_CPU_BACKEND "Build triton-shared CPU backend" ON)

set(TRITON_SHARED_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TRITON_SHARED_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include) # Tablegen'd files
include_directories(${Python3_INCLUDE_DIR})
include_directories(${pybind11_INCLUDE_DIR})
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools)

if (TRITON_SHARED_BUILD_CPU_BACKEND)
    add_triton_plugin(TritonShared ${CMAKE_CURRENT_SOURCE_DIR}/triton_shared.cc LINK_LIBS TritonSharedAnalysis TritonToLinalg TritonTilingExtIR)
    target_link_libraries(TritonShared
        PRIVATE
        MLIRAffineToStandard
        MLIRReconcileUnrealizedCasts
        MLIRSCFToControlFlow

        MLIRBufferizationTransforms
        MLIRArithTransforms
        MLIRMemRefTransforms
        MLIRSCFTransforms
        MLIRFuncTransforms
        MLIRVectorTransforms
        MLIRTensorTransforms
        MLIRTransformDialectTransforms
        MLIRTransformDialect
        MLIRTransformUtils

        MLIRArithToLLVM
        MLIRIndexToLLVM
        MLIRMathToLLVM
        MLIRComplexToLLVM
        MLIRVectorToLLVM
        MLIRVectorToLLVMPass
        MLIRFuncToLLVM
        MLIRControlFlowToLLVM
        MLIRMemRefToLLVM
        MLIRVectorToSCF
        TPtrToLLVM
        MLIRUBToLLVM

        Python3::Module
        pybind11::headers
        ${Python3_LIBRARIES}
    )
endif()

# Build and link sleef
set(SLEEF_BUILD_SHARED_LIBS ON CACHE BOOL "Build sleef shared lib" FORCE)
set(SLEEF_BUILD_DFT OFF CACHE BOOL "Don't build sleef DFT lib" FORCE)
set(SLEEF_BUILD_GNUABI_LIBS OFF CACHE BOOL "Don't build sleef gnuabi libs" FORCE)
set(SLEEF_BUILD_TESTS OFF CACHE BOOL "Don't build sleef tests" FORCE)
set(SLEEF_BUILD_SCALAR_LIB OFF CACHE BOOL "libsleefscalar will not be built." FORCE)
add_subdirectory("${CMAKE_SOURCE_DIR}/third_party/sleef" sleef)
# Override sleef's output directory with our own
set_target_properties(sleef PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
