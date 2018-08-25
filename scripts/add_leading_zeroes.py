#!/usr/bin/env python3

import os
import shutil

sourcedir = r"../resources/input/predict_input"
prefix = r""
extension = r"jpg"

files = [(f, f[f.rfind("."):], f[:f.rfind(".")].replace(prefix, "")) for f in os.listdir(sourcedir) if f.endswith(extension)]
maxlen = len(max([f[2] for f in files], key=len))

for item in files:
    zeros = maxlen - len(item[2])
    shutil.move(sourcedir + "/" + item[0], sourcedir + "/" + prefix + str(zeros * "0" + item[2]) + item[1])
