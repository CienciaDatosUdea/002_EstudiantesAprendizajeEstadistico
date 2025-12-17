#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
from datetime import date
import datetime 
#import pandas as pd
#import numpy as np
#=========================================

def fahrenheit_from(celsius):
    """Convert Celsius to Fahrenheit degrees."""
    try:
        fahrenheit = float(celsius) * 9 / 5 + 32
        fahrenheit = round(fahrenheit, 3)  # Round to three decimal places
        return str(fahrenheit)
    except ValueError:
        return "invalid input"
