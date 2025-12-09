# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 08:55:31 2021

@author: david_hds
"""
from Web_app import app

from flask import render_template, redirect, url_for
#from models import Task
from datetime import datetime
#import forms

from libs import *

def temperature_convertion(temperature):
    T = fahrenheit_from(temperature)
    print(T)
    return T
temperature_convertion(20)