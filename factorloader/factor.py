from multiprocessing import Pool
import os
import traceback
import time
import sys
sys.path.append(os.getcwd())
from stockdownload import *

class Augur(object):
    def __init__(self):
        pass

    @classmethod
    def get_stock_data(cls, start, end, code):

        data = download_domestic_index_data(code, start, end)
        
        return data

    @classmethod
    def calc_alpha(cls, path, func, data):
        try:
            t1 = time.time()
            res = func(data)
            res.to_csv(path)
            t2 = time.time()
            print(f"Factory {os.path.splitext(os.path.basename(path))[0]} time {t2-t1}")
        except Exception as e:
            print(f"generate {path} error!!!")
            # traceback.print_exc()

    @classmethod
    def get_alpha_methods(cls, self):
        return (list(filter(lambda m: m.startswith("augur") and callable(getattr(self, m)),
                            dir(self))))
    
    @classmethod
    def generate_alpha_single(cls, alpha_name, start, end, code, save, evaluation):
        # 获取计算因子所需股票数据
        stock_data = cls.get_stock_data(start, end, code)

        # 实例化因子计算的对象
        stock = cls(stock_data, save, evaluation)

        factor = getattr(cls, alpha_name)
        if factor is None:
            print('alpha name is error!!!')
            return None
        
        alpha_data = factor(stock)

        return alpha_data
            

    @classmethod
    def generate_alphas(cls, start, end, code):
        t1 = time.time()
        # 获取计算因子所需股票数据
        stock_data = cls.get_stock_data(start, end, code)

        # 实例化因子计算的对象
        stock = cls(stock_data)

        # 因子计算结果的保存路径
        path = 'factorloader/data/' + str(code)

        # 创建保存路径
        if not os.path.isdir(path):
            os.makedirs(path)

        # 创建线程池
        count = os.cpu_count()
        pool = Pool(count)

        # 获取所有因子计算的方法
        methods = cls.get_alpha_methods(cls)

        # 在线程池中计算所有alpha
        for m in methods:
            factor = getattr(cls, m)
            try:
                pool.apply_async(cls.calc_alpha(f'{path}/{m}.csv', factor, stock))
            except Exception as e:
                traceback.print_exc()

        pool.close()
        pool.join()
        t2 = time.time()
        print(f"Total time {t2-t1}")