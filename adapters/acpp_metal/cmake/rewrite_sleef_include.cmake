# Helper: copy a sleef header into the staged dir with the relative
# upstream include path rewritten so the file resolves standalone in a
# flat directory.
#
# Invoked as:
#   cmake -DINPUT=<src> -DOUTPUT=<dst> -P rewrite_sleef_include.cmake
#
# Replaces both:
#   #include "../../include/soft_fp64/soft_f64.h"   →  #include "soft_fp64/soft_f64.h"
#   #include "../../include/soft_fp64/defines.h"    →  #include "soft_fp64/defines.h"
#
# The replacement is anchored on the leading "../../include/" literal so
# we cannot accidentally eat an unrelated path component.
#
# SPDX-License-Identifier: MIT

if(NOT INPUT OR NOT OUTPUT)
    message(FATAL_ERROR "rewrite_sleef_include.cmake requires -DINPUT=<src> -DOUTPUT=<dst>")
endif()

file(READ "${INPUT}" content)
string(REPLACE "../../include/soft_fp64/" "soft_fp64/" content "${content}")
file(WRITE "${OUTPUT}" "${content}")
