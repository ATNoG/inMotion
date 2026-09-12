idf_component_register(
    SRCS "app_main.c" "jepa_model.c" "jepa_ops.c"
    INCLUDE_DIRS "."
    EMBED_FILES "lejepa_q8.bin"
)
