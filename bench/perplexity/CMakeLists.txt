# llamap-perplexity: llama.cpp's llama-perplexity, except that special tokens written in
# the text (<|im_start|>, <|eot_id|>...) are parsed as special tokens. Needed to compute
# BF16 reference logits on chat-template text. llama.cpp itself is not modified: the
# tool is rebuilt from a copy of tools/perplexity/perplexity.cpp with one call changed,
# and linked against the libraries of an existing llama.cpp build.
#
#   cmake -S bench/perplexity -B bench/perplexity/build -DLLAMA_CPP_DIR=$LLAMA_CPP_DIR
#   cmake --build bench/perplexity/build -j
cmake_minimum_required(VERSION 3.18)
project(llamap-perplexity CXX)

set(LLAMA_CPP_DIR "$ENV{LLAMA_CPP_DIR}" CACHE PATH "llama.cpp checkout, already built in build/")
set(LIBS ${LLAMA_CPP_DIR}/build/bin)
set(PPL_DIR ${LLAMA_CPP_DIR}/tools/perplexity)

file(READ ${PPL_DIR}/perplexity.cpp SRC)
set(OLD "common_tokenize(ctx, params.prompt, true)")
string(FIND "${SRC}" "${OLD}" POS)
if(POS EQUAL -1)
    message(FATAL_ERROR "tokenize call not found in ${PPL_DIR}/perplexity.cpp")
endif()
string(REPLACE "${OLD}" "common_tokenize(ctx, params.prompt, true, /* parse_special */ true)" SRC "${SRC}")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/perplexity.cpp "${SRC}")

add_executable(llamap-perplexity ${CMAKE_CURRENT_BINARY_DIR}/perplexity.cpp ${PPL_DIR}/main.cpp)
target_include_directories(llamap-perplexity PRIVATE
    ${LLAMA_CPP_DIR}/include ${LLAMA_CPP_DIR}/common ${LLAMA_CPP_DIR}/ggml/include ${LLAMA_CPP_DIR}/vendor ${PPL_DIR})
target_link_libraries(llamap-perplexity PRIVATE
    ${LIBS}/libllama-common.so ${LIBS}/libllama.so ${LIBS}/libggml.so ${LIBS}/libggml-base.so pthread)
set_target_properties(llamap-perplexity PROPERTIES BUILD_RPATH ${LIBS})
target_compile_features(llamap-perplexity PRIVATE cxx_std_17)
