from utils.SPLADE_data import create_vector
from utils.create_SH import create_sh_sk, create_sh_sk_faster, create_sh_separate
from utils.create_PSQ import create_sk_PS, create_PS_separete
from utils.create_AMSQ import create_AMSQ_separete
from utils.create_SAS import create_sas_separate
from tests.grid_search import grid_search
from tests.test_recall import recall_test, recall_test_efficient, recall_SAS
from tests.test_ip import test_ip
import time
import datetime
from utils.create_queries import create_queries
import argparse



if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(prog='super_script.py')

    parser.add_argument("-vectors", action='store_true')
    parser.add_argument("-sketchesSH", action='store_true')
    parser.add_argument("-sketchesPSQ", action='store_true')
    parser.add_argument("-recall", action='store_true')
    parser.add_argument("-ip", action='store_true')
    parser.add_argument("-q", action='store_true')
    parser.add_argument("-g", action='store_true')
    parser.add_argument("-sketchesAMSQ", action='store_true')
    parser.add_argument("-sketchesSAS", action='store_true')
    parser.add_argument("-recall_test", action='store_true')


    args = parser.parse_args()
    print(args)
    bits = 15
    sizesPSQ = [15, 33, 72, 157]

    sizesSH = [256, 512, 1024, 2048]
    labels = [256, 512, 1024, 2048]

    if(args.vectors):
        t = time.time()
        create_vector(50000)
        t_e = time.time()
        print("Time to create vectors: ")
        print(datetime.timedelta(seconds = t_e - t))

    if(args.sketchesSH):
        t = time.time()
        create_sh_separate(sizesSH, labels)
        t_e = time.time()
        print("Time to create SH sketches: ")
        print(datetime.timedelta(seconds = t_e - t))

    if(args.sketchesPSQ):
        t = time.time()
        create_PS_separete(sizesPSQ, float_size=5, key_size=13)
        t_e = time.time()
        print("Time to create PSQ sketches: ")
        print(datetime.timedelta(seconds = t_e - t))
    
    if(args.sketchesAMSQ):
        t = time.time()
        create_AMSQ_separete(sizesPSQ, float_size=5, key_size=13)
        t_e = time.time()
        print("Time to create AMSQ sketches: ")
        print(datetime.timedelta(seconds = t_e - t))

    if(args.sketchesSAS):
        t = time.time()
        create_sas_separate(sizesSH, sizesSH, 250000)
        t_e = time.time()
        print("Time to create SAS sketches: ")
        print(datetime.timedelta(seconds = t_e - t))

    if(args.q):
        t = time.time()
        create_queries(200)
        t_e = time.time()
        print("Time to create queries: ")
        print(datetime.timedelta(seconds = t_e - t))

    if(args.recall_test):
        t = time.time()
        recall_SAS(10, labels, sizesSH, 20000, [1, 5, 10, 50, 100, 500, 1000, 5000])
        t_e = time.time()
        print("Time to test recall: ")
        print(datetime.timedelta(seconds = t_e - t))

    if(args.recall):
        t = time.time()
        recall_test_efficient(20, sizesPSQ, labels, sizesSH, 1000000, [1, 5, 10, 50, 100, 500, 1000, 5000], float_quant=5, key_quant=13)
        t_e = time.time()
        print("Time to test recall: ")
        print(datetime.timedelta(seconds = t_e - t))
        
    if(args.ip):
        t = time.time()
        test_ip(sizesSH, sizesPSQ, 200)
        t_e = time.time()
        print("Time to test ip: ")
        print(datetime.timedelta(seconds = t_e - t))
    if(args.g):
        bits = []
        for i in range(1, 9):
            for j in range(8, 17):
                bits.append((i, j))
        t = time.time()
        grid_search([256, 512, 1024], bits, [50, 100, 500], 60000)
        t_e = time.time()
        print("Time to do grid_search: ")
        print(datetime.timedelta(seconds = t_e - t))