#!/shiba_env/bin/python3

import sys
import numpy as np

if __name__ == '__main__':
    for line in sys.stdin:
        print("Value " + line, end='')
        sys.stdout.flush()
