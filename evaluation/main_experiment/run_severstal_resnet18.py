#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the main thresholding experiment for severstal with resnet18."""

from evaluate_thresholding import main


if __name__ == "__main__":
    print(
        "[LAUNCH] Main experiment | dataset=severstal | backbone=resnet18",
        flush=True,
    )
    main(
        default_dataset="severstal",
        default_backbone="resnet18",
    )
