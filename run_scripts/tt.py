# from CDR.utils import visualization

import sys
print(sys.path)
from CDR import utils
import os
import CDR

# path = CDR.__path__[0]
# model_folder = f'{os.getenv("HOME")}/Projects/CDR/CDR/models/zones_train/'
model_folder = f'{CDR.__path__[0]}/models/zones_train/'
os.makedirs(model_folder, exist_ok = True)
# data path
# data_folder = '../datasets/merged_datasets/merged_txt'
data_folder = f'{os.getenv("HOME")}/Projects/CDR/datasets/merged_datasets/merged_txt'
if os.path.exists(data_folder):
    data_files = os.listdir(data_folder)
else:
    print(f"The directory {data_folder} does not exist.")

vv = False
if vv:
    print("tt vv")
# data_files = os.listdir(data_folder)

