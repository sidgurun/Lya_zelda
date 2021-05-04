Introduction
============

``Lya_zelda`` is a high-level OO Python package which aims to provide an easy and intuitive way of interacting with nearby Bluetooth Low Energy (BLE) devices (GATT servers). In essence, this package is an extension of the ``bluepy`` package created by Ian Harvey (see `here <https://github.com/IanHarvey/bluepy/>`_)


The current implementation has been developed in Python 3 and tested on a Raspberry Pi Zero W, running Raspbian 9 (stretch), but should work with Python 2.7+ (maybe with minor modifications in terms of printing and error handling) and most Debian based OSs.

Motivation
**********

As a newbie experimenter/hobbyist in the field of IoT using BLE communications, I found it pretty hard to identify a Python package which would enable one to use a Raspberry Pi (Zero W inthis case) to swiftly scan, connect to and read/write from/to a nearby BLE device (GATT server).

This package is intended to provide a quick, as well as (hopefully) easy to undestand, way of getting a simple BLE GATT client up and running, for all those out there, who, like myself, are hands-on learners and are eager to get their hands dirty from early on.

