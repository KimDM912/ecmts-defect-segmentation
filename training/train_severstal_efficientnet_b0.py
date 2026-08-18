#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the fixed severstal / efficientnet_b0 segmentation experiment."""

from train_segmentation import main


if __name__ == "__main__":
    main(default_dataset="severstal", default_backbone="efficientnet_b0")
