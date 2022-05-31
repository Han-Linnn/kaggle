#!/usr/bin/env python3
# coding: utf-8

import os
import pandas as pd

import sys
import yaml

def test():
    return 'successful!'

if __name__ == "__main__":
    command = sys.argv[1]
    functions = {
        "test": test,
    }
    output = functions[command]()
    print(yaml.dump({"output": output}))






