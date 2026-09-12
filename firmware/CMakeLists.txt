# ESP-IDF project for the lejepa int8 demo on an ESP32-D0WD-V3.
#
# The weight blob is embedded with EMBED_FILES rather than a custom partition:
# both end up in flash and both are read through the same cache, and embedding
# avoids a second address-mapping path with its own offset arithmetic. See
# docs/reports/ESP32_FEASIBILITY.md for why flash-mapping is the only option on
# a part with no PSRAM.
cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(inmotion_esp32)
