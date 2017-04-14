#!/usr/bin/env python
from flask import Flask, render_template, Response, request
import json

app = Flask("maps_quality")
app.template_filter = "templates"


@app.route("/qual_tile/<int:zoom>/<int:x>/<int:y>.svg")
def get_qual_tile(zoom, x, y):
    value = ((x+y) % 10) / 10. 
    return Response(render_template(
            "qual.svg.tmpl",
            quality="good" if value > 0.5 else "bad",
            qual_value=value),
        mimetype="image/svg+xml")


@app.route("/assess/<int:zoom>/<int:x>/<int:y>", methods=["POST"])
def assess_tile(zoom, x, y):
    app.logger.debug("data: " + request.data)
    data = json.loads(request.data)
    app.logger.debug("{} {} {} {}".format(zoom, x, y, data["good"]))
    return ''

if __name__ == "__main__":
    app.debug = True
    app.run("127.0.0.1", 3333)
