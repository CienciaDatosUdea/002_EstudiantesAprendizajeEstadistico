
from Web_app import app
from flask import render_template,jsonify
from MainExecution import *
from flask import request, Response
import forms

@app.route("/")
def index():
    return render_template("index.html")


@app.route('/health1', methods=['POST'])
def health1():
    """
    This view is aimed to verify the healthyness of the API
    :return:
    """
    return Response('{"status":"OK"}', status=200)


@app.route('/health', methods=['GET'])
def health():
    """
    This view is aimed to verify the healthyness of the API
    :return:
    """
    return Response('{"status":"OK"}', status=200)


@app.route("/conversion", methods = ["GET", "POST"])
def conversion():

    celsius = request.args.get("celsius", "")
    print("celcius", type(celsius))

    if celsius :        
        fahrenheit = temperature_convertion(celsius)        
    else :
        fahrenheit = ""
    
    print(fahrenheit)
    return render_template("conversion.html", value = fahrenheit )  #, form = form)