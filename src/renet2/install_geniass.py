import os
import subprocess


def main():
    base_dir = os.path.dirname(__file__)  
    geniass_path = os.path.join(base_dir, "tools")
    cmd = "(cd " + geniass_path + "; " + \
            "tar -xf geniass-1.00.tar.gz; cd geniass; make)"
    print('install geniass', cmd)
    subprocess.check_call(cmd, shell=True)
    print('geniass installed')

if __name__ == "__main__":
    print("install sentence splitter, only need run once")
    print("from http://www.nactem.ac.uk/y-matsu/geniass/")
    main()
