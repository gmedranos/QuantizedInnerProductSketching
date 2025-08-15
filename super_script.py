from utils.SPLADE_data import create_vector
from utils.create_SH import create_sh_sk, create_sh_sk_faster
from utils.create_PSQ import create_sk_PS
from tests.test_recall import recall_test, recall_test_efficient
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


    args = parser.parse_args()
    print(args)
    sizesPSQ = [256 // 17, 512 // 17, 1024 // 17, 2048 // 17]

    sizesSH = [256, 512, 1024, 2048]

    if(args.vectors):
        t = time.time()
        create_vector(50000)
        t_e = time.time()
        print("Time to create vectors: ")
        print(datetime.timedelta(seconds = t_e - t))
    if(args.sketchesSH):
        t = time.time()
        create_sh_sk_faster(sizesSH)
        t_e = time.time()
        print("Time to create SH sketches: ")
        print(datetime.timedelta(seconds = t_e - t))
    if(args.sketchesPSQ):
        t = time.time()
        create_sk_PS(sizesPSQ)
        t_e = time.time()
        print("Time to create PSQ sketches: ")
        print(datetime.timedelta(seconds = t_e - t))
    
    if(args.q):
        t = time.time()
        create_queries(200)
        t_e = time.time()
        print("Time to create queries: ")
        print(datetime.timedelta(seconds = t_e - t))
    if(args.recall):
        t = time.time()
        recall_test_efficient(1, sizesPSQ, sizesSH, 1000000, [50, 100, 500, 1000, 5000])
        t_e = time.time()
        print("Time to test recall: ")
        print(datetime.timedelta(seconds = t_e - t))
    if(args.ip):
        t = time.time()
        test_ip(sizesSH, sizesPSQ, 200)
        t_e = time.time()
        print("Time to test ip: ")
        print(datetime.timedelta(seconds = t_e - t))