import logging

import argparse, time
import collections
import logging
import json
import math
import os
import random
from tqdm import tqdm, trange
import sys

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

# define path
parser.add_argument('--hello', required=False, help='the path to the training file.')
args = parser.parse_args()
logger.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
logger.info("{}".format(args.hello))
print(vars(args))

for ind in trange(int(10), desc="Epoch"):
    logger.info("procedding:{}".format(ind))
    time.sleep(0.5)
