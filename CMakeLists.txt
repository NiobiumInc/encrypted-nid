cmake_minimum_required(VERSION 3.18)
project(fhe-network-monitor-sdk LANGUAGES CXX)

# ── Self-contained SDK build (customer / FHETCH-transport flow) ────────
# Build the NID stages against the niobium-client's OWN bundled OpenFHE +
# libnbfhetch (no compiler checkout). Enabled by scripts/build_task.sh:
#   cmake -S . -B build -DNIOBIUM_SDK_BUILD=ON \
#         -DCMAKE_PREFIX_PATH=<repo>/niobium-client/vendor/lib/openfhe
# Only server_standalone_sdk records/replays over the transport; the client
# stages (keygen, encrypt_mirai, decrypt_probe) are pure OpenFHE and link
# libnbfhetch passively to resolve the instrumented-OpenFHE globals.
# See NIOBIUM_INTEGRATION.md.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(NIOBIUM_SDK_BUILD "Self-contained build against the niobium-client SDK" ON)
if(NOT NIOBIUM_SDK_BUILD)
  message(FATAL_ERROR
    "This CMakeLists builds the SDK/transport flavor only (run scripts/build_task.sh).\n"
    "This repo is the SDK/transport demo; there is no local compiler build.")
endif()

set(NB_CLIENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/niobium-client")
# Overridable so CI can point at a prebuilt niobium-fhetch (matching a prebuilt
# libnbfhetch via -DNBFHETCH_LIB=...); defaults to the client's vendored copy.
set(NB_FHETCH_DIR "${NB_CLIENT_DIR}/vendor/niobium-fhetch" CACHE PATH
    "niobium-fhetch dir providing the SDK niobium/compiler.h")
set(SERVER_DIR    "${CMAKE_CURRENT_SOURCE_DIR}/Server")
set(SRC_DIR       "${CMAKE_CURRENT_SOURCE_DIR}/src")

find_package(OpenFHE CONFIG REQUIRED)      # from CMAKE_PREFIX_PATH = client's vendor/lib/openfhe
message(STATUS "SDK build: OpenFHE ${OpenFHE_INCLUDE}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenFHE_CXX_FLAGS} -Wno-error")

include_directories(
  ${OpenFHE_INCLUDE} ${OpenFHE_INCLUDE}/third-party/include
  ${OpenFHE_INCLUDE}/core ${OpenFHE_INCLUDE}/pke ${OpenFHE_INCLUDE}/binfhe
  ${SERVER_DIR} ${SRC_DIR}
  ${NB_FHETCH_DIR}/include                          # niobium/compiler.h (SDK)
  ${NB_FHETCH_DIR}/vendor/json/single_include)      # nlohmann/json (if vendored there)
link_directories(${OpenFHE_LIBDIR})

find_library(NBFHETCH_LIB NAMES nbfhetch
  PATHS "${NB_CLIENT_DIR}/build/vendor/niobium-fhetch"
        "${NB_CLIENT_DIR}/build/_deps/niobium-fhetch-build" NO_DEFAULT_PATH)
if(NOT NBFHETCH_LIB)
  message(FATAL_ERROR "libnbfhetch not found — run 'make -C niobium-client release' first")
endif()
message(STATUS "SDK build: nbfhetch ${NBFHETCH_LIB}")
get_filename_component(NBFHETCH_DIR "${NBFHETCH_LIB}" DIRECTORY)
set(CMAKE_BUILD_RPATH "${OpenFHE_LIBDIR};${NBFHETCH_DIR}")

# zlib is OPTIONAL: the client's bundled OpenFHE .so already pulls in libz at
# runtime (DT_NEEDED). Link it only if the dev package is present.
find_package(ZLIB QUIET)
set(SDK_ZLIB "")
if(ZLIB_FOUND)
  set(SDK_ZLIB ZLIB::ZLIB)
endif()

# libnbfhetch defers its OpenFHE (lbcrypto) symbols to the final link; keep
# OpenFHE in DT_NEEDED so --as-needed can't drop it.
if(UNIX AND NOT APPLE)
  add_link_options(-Wl,--no-as-needed -Wl,--disable-new-dtags)
endif()

# KitNET circuit sources, shared by decrypt_probe + the SDK server. Compiled
# per-target so each gets the right OPENFHE_CPROBES setting.
set(KITNET_SRCS ${SERVER_DIR}/Kitnet.cpp ${SERVER_DIR}/dA.cpp ${SERVER_DIR}/Packetdata.cpp)

# ── Client stages: plain OpenFHE; link nbfhetch to resolve the instrumented-
#    OpenFHE globals (they never call the recording API themselves). ────────
add_executable(keygen          ${SRC_DIR}/keygen.cpp)
add_executable(encrypt_mirai   ${SRC_DIR}/encrypt_mirai.cpp)
add_executable(decrypt_probe   ${SRC_DIR}/decrypt_probe.cpp ${KITNET_SRCS})
foreach(stage keygen encrypt_mirai decrypt_probe)
  target_link_libraries(${stage} ${OpenFHE_SHARED_LIBRARIES} ${NBFHETCH_LIB} ${SDK_ZLIB})
endforeach()

# ── SDK-linked server: record/replay via libnbfhetch -> FHETCH transport. ──
# OPENFHE_CPROBES is REQUIRED so execute_ckks()'s ct·ct EvalMult hooks fire
# (else 0 muls). Compiled with the niobium-fhetch include FIRST so
# niobium::compiler() resolves to the SDK (libnbfhetch).
add_executable(server_standalone_sdk ${SERVER_DIR}/server_standalone_sdk.cpp ${KITNET_SRCS})
target_include_directories(server_standalone_sdk BEFORE PRIVATE ${NB_FHETCH_DIR}/include)
target_compile_definitions(server_standalone_sdk PRIVATE NIOBIUM_COMPILER OPENFHE_CPROBES)
target_link_libraries(server_standalone_sdk ${OpenFHE_SHARED_LIBRARIES} ${NBFHETCH_LIB} ${SDK_ZLIB})

# Everything lands in build/ (harness/run_submission.py expects it there).
set_target_properties(keygen encrypt_mirai decrypt_probe server_standalone_sdk
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")