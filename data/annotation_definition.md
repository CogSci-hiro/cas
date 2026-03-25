# ============================================================
# Action-tier label inventory (Typologie)
# ============================================================
```
FPP_LABELS = {
    "FPP_RFC_DECL",   # Declarative request for confirmation
    "FPP_RFC_TAG",    # Tag-format request for confirmation
    "FPP_RFC_INF",    # Inference-based confirmation request
    "FPP_RFC_INT",    # Interrogative confirmation request
    "FPP_RFRC",       # Request for repair / clarification
}

SPP_LABELS = {
    "SPP_CONF_SIMP",     # Simple confirmation
    "SPP_CONF_ECHO",     # Echo confirmation
    "SPP_CONF_EXP",      # Expanded confirmation
    "SPP_CONF_OVERLAP",  # Confirmation produced in overlap
    "SPP_DISC_SIMP",     # Simple disconfirmation
    "SPP_DISC_CORR",     # Corrective disconfirmation
    "SPP_DISC_TRBL",     # Trouble-marking disconfirmation
}

MISC_LABELS = {
    "MISC_AMBIG",          # Ambiguous response
    "MISC_NO_UPT",         # No uptake
    "MISC_AUTO_D",         # Self-directed talk
    "MISC_BACK_OVERLAP",   # Backchannel overlap
    "MISC_SEQ_CLOSE",      # Sequence closing token
}


ALL_ACTION_LABELS = (
    FPP_LABELS
    | SPP_LABELS
    | MISC_LABELS
)
```