#!/usr/bin/env python
# encoding: utf-8
'''
@project : pymaf_reimp
@file    : basedict.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2022-01-21 09:42
'''

class BaseDict(dict):
    def __init__(self, json_dict):
        super(BaseDict, self).__init__()

        for (key, value) in json_dict.items():
            if type(value).__name__ == 'dict':
                self[key] = BaseDict(value)
            elif type(value).__name__ == 'list':
                to_appand = []
                for item in value:
                    if type(item).__name__ == 'dict':
                        to_appand.append(BaseDict(item))
                    else:
                        to_appand.append(item)
                self[key] = to_appand
            else:
                self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except:
            raise AttributeError(key)

    def __setattr__(self, key, value):

        if type(value).__name__ == 'dict':
            self[key] = BaseDict(value)
        elif type(value).__name__ == 'list':
            to_set = []
            for item in value:
                if type(item).__name__ == 'dict':
                    to_set.append(BaseDict(item))
                else:
                    to_set.append(item)
            self[key] = to_set
        else:
            self[key] = value