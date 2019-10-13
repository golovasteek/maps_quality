#!/usr/bin/env python
import requests
from copy import deepcopy
from scipy import ndimage
from skimage.transform import resize
import os
import numpy as np
import pickle


TOKEN=open(".mapbox_token").read().strip()
print("'{0}'".format(TOKEN))
URL_TEMPLATE="https://api.mapbox.com/v4/{map_id}/{zoom}/{x}/{y}.jpg?access_token={token}"
SCHEMA_TEMPLATE="https://api.mapbox.com/styles/v1/golovasteek/cizl1985r000o2sqjhqk0ny0m/tiles/256/{zoom}/{x}/{y}?access_token={token}"
FILE_TEMPLATE="files/{map_id}/{zoom}/{x}/{y}.png"
SETS = [
{
    "zoom": 18,
    "x": 140814,
    "y": 85975,
    "map_id": "grayschema",
    "url": SCHEMA_TEMPLATE,
    "halfspan": 50,
},
{
    "zoom": 18,
    "x": 140814,
    "y": 85975,
    "map_id": "mapbox.satellite",
    "url": URL_TEMPLATE,
    "halfspan": 50,
},
{
    "zoom": 18,
    "x": 132760,
    "y": 90286,
    "map_id": "grayschema",
    "url": SCHEMA_TEMPLATE,
    "halfspan": 50,
},
{
    "zoom": 18,
    "x": 132760,
    "y": 90286,
    "map_id": "mapbox.satellite",
    "url": URL_TEMPLATE,
    "halfspan": 50,
},
{
    "zoom": 18,
    "x": 140225,
    "y": 97486,
    "map_id": "grayschema",
    "url": SCHEMA_TEMPLATE,
    "halfspan": 50,
},
{
    "zoom": 18,
    "x": 140225,
    "y": 97486,
    "map_id": "mapbox.satellite",
    "url": URL_TEMPLATE,
    "halfspan": 50,
},
]

tile={
    "zoom": 17,
    "x": 70407,
    "y": 42987,
    "map_id": "mapbox.streets-basic"
}


def download_tile(tile):
    url = tile["url"].format(token=TOKEN, **tile)
    file_name = FILE_TEMPLATE.format(**tile)
    if os.path.exists(file_name):
        #print("Exists:", file_name)
        return
    print("Downloading: {}".format(url))
    
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(file_name, 'w+b') as f:
        resp = requests.get(url)
        assert resp.ok, resp.text
        f.write(resp.content)
    print("Downloaded:", file_name)


def download_set(set_descr):
    tile=deepcopy(set_descr)
    for x in range(set_descr['x'] - set_descr['halfspan'], set_descr['x'] + set_descr['halfspan']):
        for y in range(set_descr['y'] - set_descr['halfspan'], set_descr['y'] + set_descr['halfspan']):
            tile.update({"x": x, "y": y})
            download_tile(tile)


for s in SETS:
    download_set(s)

