#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the main thresholding experiment for kolektor with resnet34."""

from evaluate_thresholding import main


if __name__ == "__main__":
    print(
        "[LAUNCH] Main experiment | dataset=kolektor | backbone=resnet34",
        flush=True,
    )
    main(
        default_dataset="kolektor",
        default_backbone="resnet34",
    )
