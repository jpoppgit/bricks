#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Logger config.
"""

import logging

logging.basicConfig(format="%(asctime)-15s [%(levelname)s] %(module)s:%(funcName)s:%(message)s",
                    level=logging.INFO)
