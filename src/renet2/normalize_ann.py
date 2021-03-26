import sys
import os
import argparse
import pandas as pd
import time
import json 
import subprocess

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from renet2.utils.ann_utils import *


def add_d_items(d, k, v):
    if k in d:
        d[k].add(v)
    else:
        d[k] = {v}

def add_d_cnt(d, k):
    if k in d:
        d[k] += 1
    else:
        d[k] = 1


def get_id_m_dict(hit_t, _id_d, _n_d):
    _id_m = {}
    for k in _id_d:
        if k in _id_m:
            continue
        _id_s, _name = {k}, _id_d[k]
        while True:
            _tmp_id_s = set()
            for _n in _name:
                _tmp_id = _n_d[_n]
                _tmp_id_s = _tmp_id_s | _tmp_id
            if _tmp_id_s == _id_s:
                break
            else:
                _id_s = _id_s | _tmp_id_s
                for _id in _id_s:
                    _name = _name | _id_d[_id]
                #print('asd', k, _id_s)
        _tmp_hit_t = {i: hit_t[i] for i in _id_s}
        tar_id = sorted(_tmp_hit_t.items(), key=lambda x: (x[1],x[0]), reverse=True)[0][0]
        #print(sorted(_tmp_hit_t.items(), key=lambda x: x[1], reverse=True))
        for _id in _id_s:
            _id_m[_id] = tar_id
    return _id_m

def normalize_ann(config):

    in_f = config.in_f
    out_f = config.out_f
    f_anns = open(out_f, "w")

    with open(in_f, 'r') as F:
        line = F.readline()
        while True:
            if line == "\n":
                line = F.readline()
                f_anns.write("\n")
                continue
            if len(line) == 0:
                break

            # reading ann from a doc
            # gene name dict, disease id dict, etc.
            g_n_d, g_id_d, d_n_d, d_id_d = {}, {}, {}, {}
            g_hit_t, d_hit_t = {}, {}
            anns = []
            while True:
                ann = line.strip().split("\t")
                anns.append(ann[:])

                _type, _name, _id = ann[4], ann[3], ann[5]
                if _id != 'None' and _name != 'None':
                    # add ner to dict
                    # gene case
                    if _type[0] == 'G':
                        add_d_items(g_n_d, _name, _id)
                        add_d_items(g_id_d, _id, _name)
                        add_d_cnt(g_hit_t, _id)
                    else:
                        add_d_items(d_n_d, _name, _id)
                        add_d_items(d_id_d, _id, _name)
                        add_d_cnt(d_hit_t, _id)

                line = F.readline()
                if line == "\n":
                    break
            #print(anns)
            #print('---')
            #print(g_n_d, g_id_d, d_n_d, d_id_d)
            g_id_m = get_id_m_dict(g_hit_t, g_id_d, g_n_d)
            #print(g_id_m)
            #for i in g_n_d:
            #    print(i, g_n_d[i], g_id_m[next(iter(g_n_d[i]))])

            d_id_m = get_id_m_dict(d_hit_t, d_id_d, d_n_d)
            #print(d_id_m)
            #if anns[0][0] == '16781449':
            #    print(g_id_m)
            #    for i in g_n_d:
            #        print(i, g_n_d[i], g_id_m[next(iter(g_n_d[i]))])
            #    print(d_id_m)
                


            for rec in anns:
                #f_anns.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (rec[0], rec[1], rec[2], rec[3], rec[4], rec[5], rec[6]))
                #f_anns.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (rec[0], rec[1], rec[2], rec[3], rec[4], rec[5], rec[6]))
                _id = rec[5]
                if _id != 'None':
                    if rec[4][0] == 'G':
                        if _id != g_id_m[_id]:
                            _id += ';' + g_id_m[_id]
                    else:
                        if _id != d_id_m[_id]:
                            _id += ';' + d_id_m[_id]
                f_anns.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (rec[0], rec[1], rec[2], rec[3], rec[4], _id, rec[6]))

            f_anns.write("\n")

            line = F.readline()
            

            #break

    f_anns.close()







def main():
    parser = argparse.ArgumentParser(description="normalize annotation ID")

    parser.add_argument('--in_f', type=str, default=None,
            help="input annotation %(default)s")

    parser.add_argument('--out_f', type=str, default=None,
            help="output file default: %(default)s")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    normalize_ann(args)
    print('done')


if __name__ == "__main__":
    main()
