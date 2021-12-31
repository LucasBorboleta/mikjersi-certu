#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""mikjersi_make_icon_file.py makes icon file for the GUI of MIKJERSI boardgame."""


_COPYRIGHT_AND_LICENSE = """
MIKJERSI-CERTU implements a GUI and a rule engine for the MIKJERSI boardgame.

Copyright (C) 2021 Lucas Borboleta (lucas.borboleta@free.fr).

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses>.
"""


import os
from PIL import Image

_package_home = os.path.abspath(os.path.dirname(__file__))

# File path containing the icon to be displayed in the title bar of Mikjersi GUI
ICON_FILE = os.path.join(_package_home, 'pictures', 'mikjersi.ico')

# File path containing the PNG file from which ICON_FILE has to be made
PNG_FILE = os.path.join(_package_home, 'pictures', 'mikjersi-board-wo-reserve.png')

# Make icon from PNG file

print()
print("Building icon file ...")

image = Image.open(PNG_FILE)
icon_sizes = [(16,16), (32, 32), (48, 48), (64,64)]
image.save(ICON_FILE, sizes=icon_sizes)
    
print()
print("Building icon file done")

if False:
    print()
    _ = input("main: done ; press enter to terminate")

