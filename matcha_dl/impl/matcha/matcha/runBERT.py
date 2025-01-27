#!/usr/bin/env python
# coding: utf-8


from sentence_transformers import SentenceTransformer
import torch
import decimal
import sys

# create a new context for this task
ctx = decimal.Context()

# 20 digits should be enough for everyone
ctx.prec = 3

def float_to_str(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """
    d1 = ctx.create_decimal(float(f))
    return format(d1, 'f')

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
labels = sys.argv[1]
output = sys.argv[2]

f_in = open(labels, "r")
f_out = open(output, "w")

for line in f_in:
    f_out.write(line.strip())
    f_out.write("\t")
    
    embedding = model.encode(line)
    for emb in embedding:
        f_out.write(float_to_str(emb))
        f_out.write(" ")
    f_out.write("\n")
f_out.close()
