# -*- coding: utf-8 -*-

"""
Module to read the dataset and map the data to objects
"""

from radarplot.radartypes import *
import json
import os.path
import numpy as np

class CIKM (object):
    """Abstract class for reading the dataset and mapping to objets"""

    def __init__(self, filename, size=10, nlayers=4, nticks=15, mapdim=101):
        self.filename = filename
        self.nlayers = nlayers
        self.nticks = nticks
        self.mapdim = mapdim
        self.size = size
        self.radarslots = self.mapdim**2
        self.memmap = np.memmap(filename, dtype='uint8', mode='r',shape=(self.size, nticks, nlayers, mapdim, mapdim))

    def _line_ind(self, filename):
        """Yield succesive labels and positions (in bytes) of the lines in 
        the file."""
        p = 0
        with open(filename) as fileobj:
            for line in fileobj:
                label = line.split(',')[1]
                yield [float(label), p]
                p += len(line)

    def __getIdLabel(self, n, reversed):
        """Returns a tuple with the information: (id, label)
        where radarmap is a list of numbers in the format define of the spec"""
        rawmap =  self.__getFirst64(n, reversed).split(',')
        return (rawmap[0], float(rawmap[1]))

    def __getRawMap(self, n, reversed):
        """Returns a tuple with the map information: (id, label, radarmap)
        where radarmap is a list of numbers in the format define of the spec"""
        index = self.getSize()-n-1 if reversed else n
        return ('train_{}'.format(str(n+1)), 0.0, self.memmap[index].ravel())

    def _getLayerData(self, rawlayer):
        """Returns 2D array ([[row0], [row1], ..., [rowN]] from rawlayer 
        which has the format [row0row1...rowN]."""
        return rawlayer.reshape(-1, self.mapdim)

    def getSize(self):
        """Returns the number of target maps in the dataset"""
        return self.size

    def getMapDimension(self):
        return self.mapdim

    def getIdLabelRange(self, ini, end, reversed=False):
        """Yields a tuple (idmap, label) sucessively between ini and end."""
        for idx in range(ini, end):
            (idmap, label) = self.__getIdLabel(idx, reversed)
            yield (idmap, label)

    def getIdLabel(self, idx, reversed=False):
        """Returns a tuple (idmap. label) in the index idx positon in the 
        dataset"""
        return list(self.getRadarRange(idx, idx + 1, reversed))[0]

    def getAllIdLabels(self, reversed=False):
        """Yield sucessively all the tuples (idmap, label) objects in the 
        dataset"""
        for radar in self.getRadarRange(0, self.getSize(), reversed):
            yield radar
    
    def getRadarRange(self, ini, end, reversed=False):
        """Yields a Radar object sucessively between ini and end."""
        for idx in range(ini, end):
            (idmap, label, radardata) = self.__getRawMap(idx, reversed)
            radar = Radar(idmap, label)
            stackn = 0
            stack = RadarStack(radar, stackn)
            for i, l in enumerate(np.uint8(radardata).reshape(-1, self.radarslots)):
                layern = i % self.nlayers
                layer = RadarLayer(self._getLayerData(l), radar, stackn, layern)
                stack.addLayer(layer)
                if (layern == self.nlayers - 1):
                    radar.addStack(stack)
                    stackn += 1
                    stack = RadarStack(radar, stackn)
            yield radar

    def getRadar(self, idx, reversed=False):
        """Returns a Radar object in the index idx positon in the dataset"""
        return list(self.getRadarRange(idx, idx + 1, reversed))[0]

    def getAllRadars(self, reversed=False):
        """Yield sucessively all the Radars objects in the dataset"""
        for radar in self.getRadarRange(0, self.getSize(), reversed):
            yield radar
