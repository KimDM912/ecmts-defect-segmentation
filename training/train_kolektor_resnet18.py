#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the fixed kolektor / resnet18 segmentation experiment."""

from train_segmentation import main


if __name__ == "__main__":
    main(default_dataset="kolektor", default_backbone="resnet18")
