#!/usr/bin/env python
import requests
from copy import deepcopy
from scipy import ndimage
from scipy.misc import imresize
import os
import numpy as np
import cPickle as pickle


TOKEN=open(".mapbox_token").read()
URL_TEMPLATE="https://api.mapbox.com/v4/{map_id}/{zoom}/{x}/{y}.jpg?access_token={token}"
SCHEMA_TEMPLATE="https://api.mapbox.com/styles/v1/golovasteek/cizl1985r000o2sqjhqk0ny0m/tiles/256/{zoom}/{x}/{y}?access_token={token}"
FILE_TEMPLATE="files/{map_id}/{zoom}/{x}/{y}.jpg"
SETS = [
{
    "zoom": 18,
    "x": 140814,
    "y": 85975,
    "map_id": "grayschema",
    "url": SCHEMA_TEMPLATE,
    "halfspan": 10,
},
{
    "zoom": 18,
    "x": 140814,
    "y": 85975,
    "map_id": "mapbox.satellite",
    "url": URL_TEMPLATE,
    "halfspan": 10,
}]

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
        print "Exists:", file_name
        return
    
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(file_name, 'w+') as f:
        resp = requests.get(url)
        assert resp.ok
        f.write(resp.content)
    print "Downloaded:", file_name


def download_set(set_descr):
    tile=deepcopy(set_descr)
    for x in range(set_descr['x'] - set_descr['halfspan'], set_descr['x'] + set_descr['halfspan']):
        for y in range(set_descr['y'] - set_descr['halfspan'], set_descr['y'] + set_descr['halfspan']):
            tile.update({"x": x, "y": y})
            download_tile(tile)


for s in SETS:
    download_set(s)


def segments(shape, step_size=32, window_size=64):
    for x in range(0, shape[0] - window_size + 1, step_size):
        for y in range(0, shape[1] - window_size + 1, step_size):
            yield x,y , x + window_size, y + window_size


data = {
    'mapbox.satellite': {},
    'grayschema':{}
}

print "Combining images into dataset..."
for set_descr in SETS:
    tile=deepcopy(set_descr)
    for x in range(set_descr['x'] - set_descr['halfspan'], set_descr['x'] + set_descr['halfspan']):
        for y in range(set_descr['y'] - set_descr['halfspan'], set_descr['y'] + set_descr['halfspan']):
            tile.update({"x": x, "y": y})
            image = ndimage.imread(FILE_TEMPLATE.format(**tile)).astype(np.float32)
            imdata = (imresize(image, 0.5) - 256.0/2)/(256.0/2)
            data[tile['map_id']][(x, y, 0)] = imdata
            imdata = np.rot90(imdata)
            data[tile['map_id']][(x, y, 1)] = imdata
            imdata = np.rot90(imdata)
            data[tile['map_id']][(x, y, 2)] = imdata
            imdata = np.rot90(imdata)
            data[tile['map_id']][(x, y, 3)] = imdata
                

print "Creating parallel lists..."
sample = data['grayschema'].items()[0][1]
count = len(data['grayschema'].items())

satlines = np.ndarray((count, sample.shape[0], sample.shape[1], sample.shape[2]))
maplines = np.ndarray((count, sample.shape[0], sample.shape[1], 1))
print satlines.shape
print maplines.shape
i = 0
for index, img in data['grayschema'].iteritems():
    satimg = data['mapbox.satellite'][index]
    satlines[i] = satimg[:,:,:].reshape(( sample.shape[0], sample.shape[1], sample.shape[2])).astype(np.float32)
    maplines[i] = img[:,:,0].reshape(( sample.shape[0], sample.shape[1], 1)).astype(np.float32)
    i += 1

print "Saving intermediate results..."
with open('satlines.npy', 'w+') as f:
    np.save(f, satlines)
with open('maplines.npy', 'w+') as f:
    np.save(f, maplines)


with open('satlines.npy') as f:
    satlines = np.load(f)
with open('maplines.npy') as f:
    maplines = np.load(f)


print "Combining good and bad pairs..."
count = satlines.shape[0]
good_pairs = np.concatenate((satlines, maplines), axis=3)
good_labels = np.concatenate((np.ones((count,1)), np.zeros((count,1))), axis=1)
np.random.shuffle(maplines)
bad_pairs_1 = np.concatenate((satlines, maplines), axis=3)
bad_labels = np.concatenate((np.zeros((count,1)), np.ones((count,1))), axis=1)


data_lines = np.concatenate((good_pairs, bad_pairs_1))
labels = np.concatenate((good_labels, bad_labels))

print "Shuffling..."
indicies = range(count*2)
np.random.shuffle(indicies)
data_lines = data_lines[indicies]
labels = labels[indicies]

print "Saving..."
data_lines = data_lines.astype(np.float32)
labels = labels.astype(np.float32)
with open('datalines.npy', 'w+') as f:
    np.save(f, data_lines)
with open('labels.npy', 'w+') as f:
    np.save(f, labels)


with open('datalines_500.npy', 'w+') as f:
    np.save(f, data_lines[:500])
with open('labels_500.npy', 'w+') as f:
    np.save(f, labels[:500])

