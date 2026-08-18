#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the main thresholding experiment for magnetic with efficientnet_b0."""

from evaluate_thresholding import main


if __name__ == "__main__":
    print(
        "[LAUNCH] Main experiment | dataset=magnetic | backbone=efficientnet_b0",
        flush=True,
    )
    main(
        default_dataset="magnetic",
        default_backbone="efficientnet_b0",
    )
