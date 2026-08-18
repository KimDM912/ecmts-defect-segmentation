#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the main thresholding experiment for severstal with efficientnet_b0."""

from evaluate_thresholding import main


if __name__ == "__main__":
    print(
        "[LAUNCH] Main experiment | dataset=severstal | backbone=efficientnet_b0",
        flush=True,
    )
    main(
        default_dataset="severstal",
        default_backbone="efficientnet_b0",
    )
