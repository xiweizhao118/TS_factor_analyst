'''
    Inputs
    ------
    zz500 index
    columns: date      symbol      open      high       low     close       volume

    Returns
    -------
    factor index
    columns: date      factor
'''
import akshare as ak
import numpy as np
import pandas as pd
from . import mymodel as model
from .factor import Augur
# import mymodel as model
# from factor import Augur
import math
import sys
import os
sys.path.append(os.getcwd())
from stockdownload import *
import utils as tools

SAVE_PATH = 'factorloader/data'


class Augurs(Augur):
    def __init__(self, df, save=False, evaluation=False):
        self.df = df
        self.evaluation = evaluation
        self.save = save

    def augur_0001(self):
        '''
        Turbulence Index
        ref: https://zhuanlan.zhihu.com/p/629602283?utm_medium=social&utm_oi=38417056399360&utm_psn=1641600863545217024&utm_source=wechat_session

        Method
        ------
        turbulence_t = 1/n * (r_t - \miu_{t'})^T \Sigma_{t'}^{-1} (r_t - \miu_{t'})
        where:  n denotes the assets number;
                r_t denotes return rate vector of n assets during certain period;
                \miu_{t'} denotes the average return rate of n assets during certain past period;
                \Sigma_{t'} denotes the return rate covariance matrix of n assets during certain past period.

        Params
        ------
        zz500 index
        columns: date      symbol      open      high       low     close       volume

        Returns
        -------
        turbulence index
        columns: date      turbulence

        '''

        ddf = self.df.pivot(index='date', columns='symbol', values='close')
        date = self.df.date.unique()

        start = 25
        turbulence_index = [0]*start

        for i in range(start, len(date)):
            current_price = ddf[ddf.index == date[i]]
            hist_price = ddf[ddf.index <= date[i]]
            cov_temp = hist_price.cov()
            current_temp = (current_price - np.mean(hist_price, axis=0))
            turbulence_temp = current_temp.values.dot(
                np.linalg.inv(cov_temp)).dot(current_temp.values.T)
            turbulence_index.append(turbulence_temp.squeeze().tolist())

        factor = pd.DataFrame({'date': ddf.index,
                            'factor': turbulence_index})

        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)

        return factor


    def augur_0002(self):
        '''
        Skew Index: 
        The simplest type of book feature is called the skew, which is the imbalance between resting bid depth and resting ask depth.

        Method
        ------
        skew = log(v_2) - log(v_1)
        where:  v_2 denotes the resting bid depth;
                v_1 denotes the resting ask depth.

        '''
        skew = np.log(self.df['close'] / self.df['open']).values
        factor = pd.DataFrame({'date': self.df['date'],
                            'factor': skew})

        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)

        return factor


    def augur_0003(self):
        '''
        Imbalance Index:
        A variation of the skew is called a book imbalance.

        Method
        ------
        imbalance = (v_2 - v_1) / (v_2 + v_1)
        where:  v_2 denotes the resting bid depth;
                v_1 denotes the resting ask depth.

        '''
        imbalance = ((self.df['close'] - self.df['open']) /
                    (self.df['close'] + self.df['open'])).values
        factor = pd.DataFrame({'date': self.df['date'],
                            'factor': imbalance})

        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)

        return factor  

    def augur_0004(self):
        '''
        Domestic Index:
        股票-国内市场-大盘股: 沪深300

        '''
        code = "000300"
        start_time = list(self.df['date'])[0]
        end_time = list(self.df['date'])[-1]
        data = download_domestic_index_data(code, start_time, end_time)

        factor = pd.DataFrame({'date': data['date'],
                            'factor': data['close']})
        
        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0005(self):
        '''
        Domestic Index:
        股票-国内市场-小盘股: 中证1000

        '''
        code = "000852"
        start_time = list(self.df['date'])[0]
        end_time = list(self.df['date'])[-1]
        data = download_domestic_index_data(code, start_time, end_time)

        factor = pd.DataFrame({'date': data['date'],
                            'factor': data['close']})
        
        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0006(self):
        '''
        Domestic Index:
        债券-国内债券-国债：国债指数（东方财富）

        '''
        code = "000012"
        start_time = list(self.df['date'])[0]
        end_time = list(self.df['date'])[-1]
        data = download_domestic_index_data(code, start_time, end_time)

        factor = pd.DataFrame({'date': data['date'],
                            'factor': data['close']})
        
        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0007(self):
        '''
        Domestic Index:
        债券-国内债券-企业债：企债指数（东方财富）

        '''
        code = "000013"
        start_time = list(self.df['date'])[0]
        end_time = list(self.df['date'])[-1]
        data = download_domestic_index_data(code, start_time, end_time)

        factor = pd.DataFrame({'date': data['date'],
                            'factor': data['close']})
        
        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0008(self):
        '''
        Domestic Index:
        商品-农产品类-农产品：上证农产品指数

        '''
        code = "000949"
        start_time = list(self.df['date'])[0]
        end_time = list(self.df['date'])[-1]
        data = download_domestic_index_data(code, start_time, end_time)

        factor = pd.DataFrame({'date': data['date'],
                            'factor': data['close']})
        
        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0009(self):
        '''
        Domestic Index:
        其他另类投资-房地产：上证地产指数

        '''
        code = "000006"
        start_time = list(self.df['date'])[0]
        end_time = list(self.df['date'])[-1]
        data = download_domestic_index_data(code, start_time, end_time)

        factor = pd.DataFrame({'date': data['date'],
                            'factor': data['close']})
        
        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0010(self):
        '''
        Domestic Index:
        其他另类投资-医药：上证医药指数

        '''
        code = "000037"
        start_time = list(self.df['date'])[0]
        end_time = list(self.df['date'])[-1]
        data = download_domestic_index_data(code, start_time, end_time)

        factor = pd.DataFrame({'date': data['date'],
                            'factor': data['close']})
        
        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0011(self):
        '''
        Domestic Industry Data:
        商品-金属类-贵金属：东方财富沪京深行业板贵金属

        '''
        code = "贵金属"
        start_time = str(self.df['date'][0])[:10]
        end_time = str(self.df['date'].tolist()[-1])[:10]
        data = download_domestic_industry_data(code, start_time.replace('-', ''), end_time.replace('-', ''))

        factor = pd.DataFrame({'date': data['date'],
                            'factor': data['close']})
        
        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0012(self):
        '''
        Domestic Industry Data:
        商品-金属类-有色金属：东方财富沪京深行业板有色金属

        '''
        code = "有色金属"
        start_time = str(self.df['date'][0])[:10]
        end_time = str(self.df['date'].tolist()[-1])[:10]
        data = download_domestic_industry_data(code, start_time.replace('-', ''), end_time.replace('-', ''))

        factor = pd.DataFrame({'date': data['date'],
                            'factor': data['close']})
        
        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0013(self):
        '''
        Domestic Industry Data:
        商品-金属类-能源金属：东方财富沪京深行业板能源金属

        '''
        code = "能源金属"
        start_time = str(self.df['date'][0])[:10]
        end_time = str(self.df['date'].tolist()[-1])[:10]
        data = download_domestic_industry_data(code, start_time.replace('-', ''), end_time.replace('-', ''))

        factor = pd.DataFrame({'date': data['date'],
                            'factor': data['close']})
        
        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0014(self):
        '''
        Domestic Industry Data:
        商品-能源化工类-化肥行业：东方财富沪京深行业板化肥行业

        '''
        code = "化肥行业"
        start_time = str(self.df['date'][0])[:10]
        end_time = str(self.df['date'].tolist()[-1])[:10]
        data = download_domestic_industry_data(code, start_time.replace('-', ''), end_time.replace('-', ''))

        factor = pd.DataFrame({'date': data['date'],
                            'factor': data['close']})
        
        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0015(self):
        '''
        Domestic Industry Data:
        商品-能源化工类-化纤行业：东方财富沪京深行业板化纤行业

        '''
        code = "化纤行业"
        start_time = str(self.df['date'][0])[:10]
        end_time = str(self.df['date'].tolist()[-1])[:10]
        data = download_domestic_industry_data(code, start_time.replace('-', ''), end_time.replace('-', ''))

        factor = pd.DataFrame({'date': data['date'],
                            'factor': data['close']})
        
        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0016(self):
        '''
        Domestic Industry Data:
        商品-能源化工类-化学原料：东方财富沪京深行业板化学原料

        '''
        code = "化学原料"
        start_time = str(self.df['date'][0])[:10]
        end_time = str(self.df['date'].tolist()[-1])[:10]
        data = download_domestic_industry_data(code, start_time.replace('-', ''), end_time.replace('-', ''))

        factor = pd.DataFrame({'date': data['date'],
                            'factor': data['close']})
        
        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0017(self):
        '''
        Domestic Industry Data:
        商品-能源化工类-橡胶制品：东方财富沪京深行业板橡胶制品

        '''
        code = "橡胶制品"
        start_time = str(self.df['date'][0])[:10]
        end_time = str(self.df['date'].tolist()[-1])[:10]
        data = download_domestic_industry_data(code, start_time.replace('-', ''), end_time.replace('-', ''))

        factor = pd.DataFrame({'date': data['date'],
                            'factor': data['close']})
        
        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0018(self):
        '''
        Domestic Industry Data:
        商品-能源化工类-燃气：东方财富沪京深行业板燃气

        '''
        code = "燃气"
        start_time = str(self.df['date'][0])[:10]
        end_time = str(self.df['date'].tolist()[-1])[:10]
        data = download_domestic_industry_data(code, start_time.replace('-', ''), end_time.replace('-', ''))

        factor = pd.DataFrame({'date': data['date'],
                            'factor': data['close']})
        
        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0019(self):
        '''
        Domestic Industry Data:
        商品-能源化工类-石油行业：东方财富沪京深行业板石油行业

        '''
        code = "石油行业"
        start_time = str(self.df['date'][0])[:10]
        end_time = str(self.df['date'].tolist()[-1])[:10]
        data = download_domestic_industry_data(code, start_time.replace('-', ''), end_time.replace('-', ''))

        factor = pd.DataFrame({'date': data['date'],
                            'factor': data['close']})
        
        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0020(self):
        '''
        Abroad Index:
        股票-海外成熟市场-港股：恒生指数

        '''
        code = "^HSI"
        start_time = list(self.df['date'])[0]
        end_time = list(self.df['date'])[-1]
        data = download_abroad_index_data(code, start_time, end_time).reset_index(drop=True)

        factor = pd.DataFrame({'date': data['date'],
                            'factor': data['close']})
        
        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0021(self):
        '''
        Abroad Index:
        股票-海外成熟市场-美股: 标普500

        '''
        code = "^GSPC"
        start_time = list(self.df['date'])[0]
        end_time = list(self.df['date'])[-1]
        data = download_abroad_index_data(code, start_time, end_time).reset_index(drop=True)

        factor = pd.DataFrame({'date': data['date'],
                            'factor': data['close']})
        
        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0022(self):
        '''
        Abroad Index:
        股票-海外新兴市场: iShares MSCI新兴市场ETF

        '''
        code = "EEM"
        start_time = list(self.df['date'])[0]
        end_time = list(self.df['date'])[-1]
        data = download_abroad_index_data(code, start_time, end_time).reset_index(drop=True)

        factor = pd.DataFrame({'date': data['date'],
                            'factor': data['close']})
        
        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0023(self):
        '''
        Abroad Index:
        债券-海外债券-美元债：上市指数基金美国债券(无货币对冲)

        '''
        code = "1486.T"
        start_time = list(self.df['date'])[0]
        end_time = list(self.df['date'])[-1]
        data = download_abroad_index_data(code, start_time, end_time).reset_index(drop=True)

        factor = pd.DataFrame({'date': data['date'],
                            'factor': data['close']})
        
        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0024(self):
        '''
        Abroad Index:
        货币-本币-人民币：人民币汇率指数

        '''
        code = "CNY=X"
        start_time = list(self.df['date'])[0]
        end_time = list(self.df['date'])[-1]
        data = download_abroad_index_data(code, start_time, end_time).reset_index(drop=True)

        factor = pd.DataFrame({'date': data['date'],
                            'factor': data['close']})
        
        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0025(self):
        '''
        Abroad Index:
        货币-外汇-美元：美元指数

        '''
        code = "DX-Y.NYB"
        start_time = list(self.df['date'])[0]
        end_time = list(self.df['date'])[-1]
        data = download_abroad_index_data(code, start_time, end_time).reset_index(drop=True)

        factor = pd.DataFrame({'date': data['date'],
                            'factor': data['close']})
        
        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0026(self):
        '''
        Note: data is updated monthly
        官方制造业PMI - https://datacenter.jin10.com/reportType/dc_chinese_manufacturing_pmi
        '''

        start_time = list(self.df['date'])[0]
        end_time = list(self.df['date'])[-1]

        factor = ak.macro_china_pmi_yearly()
        factor = pd.DataFrame(tools.roughly_round_to_yearmonth(factor))
        factor = factor[(factor.index >= start_time) & (factor.index <= end_time)]
        factor = factor.reset_index()
        factor.columns = ['date','factor']

        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0027(self):
        '''
        Note: data is updated monthly
        财新制造业PMI终值 - https://datacenter.jin10.com/reportType/dc_chinese_caixin_manufacturing_pmi
        '''

        start_time = list(self.df['date'])[0]
        end_time = list(self.df['date'])[-1]

        factor = ak.macro_china_cx_pmi_yearly()
        factor = pd.DataFrame(tools.roughly_round_to_yearmonth(factor))
        factor = factor[(factor.index >= start_time) & (factor.index <= end_time)]
        factor = factor.reset_index()
        factor.columns = ['date','factor']


        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0028(self):
        '''
        Note: data is updated monthly
        中国官方非制造业PMI - https://datacenter.jin10.com/reportType/dc_chinese_non_manufacturing_pmi
        '''

        start_time = list(self.df['date'])[0]
        end_time = list(self.df['date'])[-1]

        factor = ak.macro_china_non_man_pmi()
        factor = pd.DataFrame(tools.roughly_round_to_yearmonth(factor))
        factor = factor[(factor.index >= start_time) & (factor.index <= end_time)]
        factor = factor.reset_index()
        factor.columns = ['date','factor']

        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0029(self):
        '''
        Note: data is updated monthly
        采购经理人指数 - http://data.eastmoney.com/cjsj/pmi.html
        original columns: 月份 制造业-指数 制造业-同比增长 非制造业-指数 非制造业-同比增长
        default return: 制造业-指数
        '''

        start_time = list(self.df['date'])[0]
        end_time = list(self.df['date'])[-1]

        factor = ak.macro_china_pmi()
        factor['月份'] = pd.to_datetime([i.replace('年','-').replace('月份','-') + "28" for i in factor['月份'].tolist()])
        factor = factor.rename(columns={'月份':'date'})
        factor = factor[(factor['date'] >= start_time) & (factor['date'] <= end_time)]
        factor = pd.DataFrame({'date': factor['date'],
                            'factor': factor['制造业-指数']})
        factor = factor.reindex(index=factor.index[::-1]).reset_index().drop(columns = 'index')

        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0030(self):
        '''
        Note: data is updated monthly
        社会融资规模增量统计 - http://data.mofcom.gov.cn/gnmy/shrzgm.shtml
        original columns: 月份	社会融资规模增量    其中-人民币贷款	其中-委托贷款外币贷款	其中-委托贷款	其中-信托贷款	其中-未贴现银行承兑汇票	其中-企业债券	其中-非金融企业境内股票融资
        default return: 社会融资规模增量
        '''

        start_time = list(self.df['date'])[0]
        end_time = list(self.df['date'])[-1]

        factor = ak.macro_china_shrzgm()

        mm = []
        for m in factor['月份'].tolist():
            m = list(m)
            m.insert(4,'-')
            m = "".join(m)+'-28'
            mm.append(m)
        factor['月份'] = pd.to_datetime(mm)
        factor = factor.rename(columns={'月份':'date'})
        factor = factor[(factor['date'] >= start_time) & (factor['date'] <= end_time)]
        factor = pd.DataFrame({'date': factor['date'],
                            'factor': factor['社会融资规模增量']})
        factor = factor.reset_index().drop(columns = 'index')

        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0031(self):
        '''
        Note: data is updated monthly
        企业商品价格指数 - http://data.eastmoney.com/cjsj/qyspjg.html
        original columns: 月份	总指数-指数值	总指数-同比增长	总指数-环比增长	农产品-指数值	农产品-同比增长	农产品-环比增长	矿产品-指数值	矿产品-同比增长	矿产品-环比增长	煤油电-指数值	煤油电-同比增长	煤油电-环比增长
        default return: 总指数-指数值
        
        '''

        start_time = list(self.df['date'])[0]
        end_time = list(self.df['date'])[-1]

        factor = ak.macro_china_qyspjg()
        factor['月份'] = pd.to_datetime([i.replace('年','-').replace('月份','-') + "28" for i in factor['月份'].tolist()])
        factor = factor.rename(columns={'月份':'date'})
        factor = factor[(factor['date'] >= start_time) & (factor['date'] <= end_time)]
        factor = pd.DataFrame({'date': factor['date'],
                            'factor': factor['总指数-指数值']})
        factor = factor.reset_index().drop(columns = 'index')

        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0032(self):
        '''
        Note: data is updated monthly
        外汇储备(亿美元) - https://datacenter.jin10.com/reportType/dc_chinese_fx_reserves

        '''

        start_time = list(self.df['date'])[0]
        end_time = list(self.df['date'])[-1]

        factor = ak.macro_china_fx_reserves_yearly()
        factor = pd.DataFrame(factor)
        factor = factor[(factor.index >= start_time) &(factor.index <= end_time)]
        factor = factor.reset_index()
        factor.columns = ['date','factor']

        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0033(self):
        '''
        Note: data is updated seasonally
        国内生产总值 - http://data.eastmoney.com/cjsj/gdp.html
        original columns: 季度	国内生产总值-绝对值	国内生产总值-同比增长	第一产业-绝对值	第一产业-同比增长	第二产业-绝对值	第二产业-同比增长	第三产业-绝对值	第三产业-同比增长
        default return: 国内生产总值-绝对值
        '''

        start_time = list(self.df['date'])[0]
        end_time = list(self.df['date'])[-1]

        factor = ak.macro_china_gdp()

        ss = []
        for s in factor['季度']:
            s = s.replace('年第1','').replace('季度','')
            if len(s)==4:
                ss.append(s+'-03-31')
            elif s[-1] == '2':
                s = s[:-1]
                ss.append(s+'06-30')
            elif s[-1] == '3':
                s = s[:-1]
                ss.append(s+'09-30')
            elif s[-1] == '4':
                s = s[:-1]
                ss.append(s+'12-31')
        
        factor['季度'] = pd.to_datetime(ss)
        factor = factor.rename(columns={'季度':'date'})
        factor = factor[(factor['date'] >= start_time) & (factor['date'] <= end_time)]
        factor = pd.DataFrame({'date': factor['date'],
                            'factor': factor['国内生产总值-绝对值']})
        factor = factor.reindex(index=factor.index[::-1]).reset_index().drop(columns = 'index')

        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0034(self):
        '''
        Note: data is updated yearly
        中国宏观杠杆率 -  http://114.115.232.154:8080/
        original columns: 年份	居民部门	非金融企业部门	政府部门	中央政府	地方政府	实体经济部门	金融部门资产方	金融部门负债方
        default return: 政府部门
        '''

        start_time = list(self.df['date'])[0]
        end_time = list(self.df['date'])[-1]

        factor = ak.macro_cnbs()
        
        factor['年份'] = pd.to_datetime([i + "-28" for i in factor['年份'].tolist()])
        factor = factor.rename(columns={'年份':'date'})
        factor = factor[(factor['date'] >= start_time) & (factor['date'] <= end_time)]
        factor = pd.DataFrame({'date': factor['date'],
                            'factor': factor['政府部门']})
        factor = factor.reset_index().drop(columns = 'index')

        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0035(self):
        '''
        Note: data is updated monthly
        美联储利率决议报告 - https://datacenter.jin10.com/reportType/dc_usa_interest_rate_decision
        original columns: 商品	日期	今值	预测值	前值
        default return: 今值
        '''

        start_time = list(self.df['date'])[0]
        end_time = list(self.df['date'])[-1]

        factor = ak.macro_bank_usa_interest_rate()
        
        factor['日期'] = pd.to_datetime(factor['日期'])
        factor = factor.rename(columns={'日期':'date'})
        factor = factor[(factor['date'] >= start_time) & (factor['date'] <= end_time)]
        factor = pd.DataFrame({'date': factor['date'],
                            'factor': factor['今值']})
        factor = factor.reset_index().drop(columns = 'index')

        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0036(self):
        '''
        Note: data is updated monthly
        中国人民银行利率报告 - https://datacenter.jin10.com/reportType/dc_china_interest_rate_decision
        original columns: 商品	日期	今值	预测值	前值
        default return: 今值
        '''

        start_time = list(self.df['date'])[0]
        end_time = list(self.df['date'])[-1]

        factor = ak.macro_bank_china_interest_rate()
        
        factor['日期'] = pd.to_datetime(factor['日期'])
        factor = factor.rename(columns={'日期':'date'})
        factor = factor[(factor['date'] >= start_time) & (factor['date'] <= end_time)]
        factor = pd.DataFrame({'date': factor['date'],
                            'factor': factor['今值']})
        factor = factor.reset_index().drop(columns = 'index')

        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0037(self):
        '''
        Note: data is updated daily
        中美国债收益率 - http://data.eastmoney.com/cjsj/zmgzsyl.html
        original columns: 日期	中国国债收益率2年	中国国债收益率5年	中国国债收益率10年	中国国债收益率30年	中国国债收益率10年-2年	中国GDP年增率	美国国债收益率2年	美国国债收益率5年	美国国债收益率10年	美国国债收益率30年	美国国债收益率10年-2年	美国GDP年增率
        default return: 中国国债收益率2年
        '''

        start_time = list(self.df['date'])[0]
        end_time = list(self.df['date'])[-1]

        factor = ak.bond_zh_us_rate()
        
        factor['日期'] = pd.to_datetime(factor['日期'])
        factor = factor.rename(columns={'日期':'date'})
        factor = factor[(factor['date'] >= start_time) & (factor['date'] <= end_time)]
        factor = pd.DataFrame({'date': factor['date'],
                            'factor': factor['中国国债收益率2年']})
        factor = factor.reset_index().drop(columns = 'index')

        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0038(self):
        '''
        Note: data is updated daily
        国债及其他债券收益率曲线 - http://yield.chinabond.com.cn/cbweb-pbc-web/pbc/historyQuery?startDate=2019-02-07&endDate=2020-02-04&gjqx=0&qxId=ycqx&locale=cn_ZH
        original columns:name	date	3mo	6mo	1yr	3yr	5yr	7yr	10yr	30yr
        债券种类: 中债中短期票据收益率曲线(AAA)
        default return: 3mo
        '''

        start_time = list(self.df['date'])[0]
        end_time = list(self.df['date'])[-1]

        factor = ak.bond_china_yield(start_date=str(start_time).replace('-','')[:-9], end_date=str(end_time).replace('-','')[:-9])
        factor.columns = ["name", "date", "3mo", "6mo", "1yr", "3yr", "5yr", "7yr", "10yr", "30yr"]
        factor = factor.loc[factor['name']=="中债中短期票据收益率曲线(AAA)"]
        
        factor['date'] = pd.to_datetime(factor['date'])
        factor = pd.DataFrame({'date': factor['date'],
                            'factor': factor['3mo']})
        factor = factor.reset_index().drop(columns = 'index')

        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0039(self):
        '''
        Note: data is updated daily
        国债及其他债券收益率曲线 - http://yield.chinabond.com.cn/cbweb-pbc-web/pbc/historyQuery?startDate=2019-02-07&endDate=2020-02-04&gjqx=0&qxId=ycqx&locale=cn_ZH
        original columns:name	date	3mo	6mo	1yr	3yr	5yr	7yr	10yr	30yr
        债券种类: 中债商业银行普通债收益率曲线(AAA)
        default return: 3mo
        '''

        start_time = list(self.df['date'])[0]
        end_time = list(self.df['date'])[-1]

        factor = ak.bond_china_yield(start_date=str(start_time).replace('-','')[:-9], end_date=str(end_time).replace('-','')[:-9])
        factor.columns = ["name", "date", "3mo", "6mo", "1yr", "3yr", "5yr", "7yr", "10yr", "30yr"]
        factor = factor.loc[factor['name']=="中债商业银行普通债收益率曲线(AAA)"]
        
        factor['date'] = pd.to_datetime(factor['date'])
        factor = pd.DataFrame({'date': factor['date'],
                            'factor': factor['3mo']})
        factor = factor.reset_index().drop(columns = 'index')

        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    def augur_0040(self):
        '''
        Note: data is updated daily
        国债及其他债券收益率曲线 - http://yield.chinabond.com.cn/cbweb-pbc-web/pbc/historyQuery?startDate=2019-02-07&endDate=2020-02-04&gjqx=0&qxId=ycqx&locale=cn_ZH
        original columns:name	date	3mo	6mo	1yr	3yr	5yr	7yr	10yr	30yr
        债券种类: 中债国债收益率曲线
        default return: 3mo
        '''

        start_time = list(self.df['date'])[0]
        end_time = list(self.df['date'])[-1]

        factor = ak.bond_china_yield(start_date=str(start_time).replace('-','')[:-9], end_date=str(end_time).replace('-','')[:-9])
        factor.columns = ["name", "date", "3mo", "6mo", "1yr", "3yr", "5yr", "7yr", "10yr", "30yr"]
        factor = factor.loc[factor['name']=="中债国债收益率曲线"]
        
        factor['date'] = pd.to_datetime(factor['date'])
        factor = pd.DataFrame({'date': factor['date'],
                            'factor': factor['3mo']})
        factor = factor.reset_index().drop(columns = 'index')

        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)
        
        return factor

    # def augur_0041(self):
    #     '''
    #     Basis factor
        
    #     Method
    #     ------
    #     factor = S * exp(r * (T-t)) - S
    #     where:  r denotes yearly return rate of china bond;
    #             T denotes the contract expiration year;
    #             t denotes the current year;
    #             S denotes the current price vector.

    #     Assumption:
    #     -----------
    #     T-t = 252
        
    #     '''

    #     T = 252
    #     ret = self.augur_0040(save=False)
    #     ret = ret['factor'] / 100
    #     f = self.df['close'] * np.exp(ret * T / 252)
    #     factor = [0] * T
    #     factor = factor + [self.df['close'][i+T] - f[i] for i in range(len(self.df)-T)]

    #     factor = pd.DataFrame({'date': self.df['date'],
    #                         'factor': factor})

    #     if self.save:
    #         tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)

    #     return factor

    def augur_0042(self):
        '''
        Basis Momentum factor

        ref: https://www.htfc.com/wz_upload/png_upload/20210714/1626254361452e581a9.pdf

        Method:
        -------
        BM = \prod_{t-R-1}^{t}(1+f_s^{T1-1} - f_{s-1}^T1) - \prod_{t-R-1}^{t}(1+f_s^{T2-1} - f_{s-1}^T2)
        where:  R denotes momemtum calculation period;
                t denotes the current date;
                T1, T2 denotes the two different transaction period lengths;
                f_s^T denotes futures price log value under the certain contract.

        '''
        
        T1 = 252
        T2 = 378
        R = 126

        def f_s_t(s, T):
            ret = 0.01
            f = s * np.exp(ret * T / 252)
            return np.log(f)

        BM = [0] * R
        for i in range(R, len(self.df)):
            tmp1 = 1
            tmp2 = 1
            for j in range(R):
                tmp1 = tmp1 * (1 + f_s_t(self.df['close'][i-j],T1-1) - f_s_t(self.df['close'][i-j-1],T1))
                tmp2 = tmp2 * (1 + f_s_t(self.df['close'][i-j],T2-1) - f_s_t(self.df['close'][i-j-1],T2))
            BM.append(tmp1 - tmp2)

        factor = pd.DataFrame({'date': self.df['date'],
                            'factor': BM})

        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)

        return factor

    def augur_0043(self):
        '''
        standard model: LSTM

        Method:
        -------
        Predict the future return based on past ten days data.

        '''

        # Step1: model parameters
        seq_length = 10
        test_size = 0.2

        LSTM_parameters = {
            'input_size': 1,
            'hidden_size': 1,
            'num_layers': 1,
            'num_classes': 1
        }
        train_parameters = {
            'num_epochs': 2000,
            'learning_rate': 0.01
        }

        # Step2: model initialization
        LSTMEngine = model.LSTMEngine(self.df, seq_length, test_size)

        # Step3: train model
        lstm = LSTMEngine.train(LSTM_parameters, train_parameters)

        # Step4: predict and evaluation
        data_predict = LSTMEngine.predict(lstm, self.evaluation)

        # Step5: calculate daily return
        pred_daily_return= pd.Series(np.log(data_predict)).diff(1).fillna(0)

        factor = pd.DataFrame({'date': self.df['date'],
                            'factor': pred_daily_return})
        
        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)

        return factor

    def augur_0044(self):
        '''
        standard model: XGBoost
        ref: https://github.com/a-bean-sprout/XGBoost_StockPricePredict

        Method:
        -------
        Predict the future return based on past ten days data.

        '''

        # Step1: set model parameters
        test_size = 0.2                # proportion of dataset to be used as test set
        N = 3                          # for feature at day t, we use lags from t-1, t-2, ..., t-N as features

        GridSearchCV_parameters={
            'n_estimators':[90],
            'max_depth':[7],
            'learning_rate': [0.3],
            'min_child_weight':range(5, 21, 1),
        }

        XGB_parameters={
            'seed': 100,
            'n_estimators': 100,
            'max_depth': 3,
            'eval_metric': 'rmse',
            'learning_rate': 0.1,
            'min_child_weight':1,
            'subsample':1,
            'colsample_bytree':1,
            'colsample_bylevel':1,
            'gamma':0
        }

        # Step2: train model
        Engine = model.XGBEngine(self.df, test_size, N)

        # Step3: predict and evaluation
        data_predict = Engine.XGBTrain(GridSearchCV_parameters, XGB_parameters, self.evaluation)

        # Step4: calculate daily return
        pred_daily_return= pd.Series(np.log(data_predict)).diff(1).fillna(0)

        factor = pd.DataFrame({'date': self.df['date'],
                            'factor': pred_daily_return})
        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)

        return factor

    def augur_0045(self):
        '''
        standard model: LPPLS
        ref: https://github.com/Boulder-Investment-Technologies/lppls

        Method:
        -------
        Predict the termination of the bubble based on past data.

        '''

        # convert time to ordinal
        time = [pd.Timestamp.toordinal(t1) for t1 in self.df['date']]


        # create list of observation data
        price = np.log(self.df['close'].values)

        # create observations array (expected format for LPPLS observations)
        observations = np.array([time, price])

        # set the max number for searches to perform before giving-up
        # the literature suggests 25
        MAX_SEARCHES = 25

        # instantiate a new LPPLS model with the Nasdaq Dot-com bubble dataset
        lppls_model = model.LPPLS(observations=observations)

        # compute the confidence indicator
        res = lppls_model.mp_compute_nested_fits(
            workers=8,
            window_size=64, 
            smallest_window_size=10, 
            outer_increment=1, 
            inner_increment=5, 
            max_searches=25,
            # filter_conditions_config={} # not implemented in 0.6.x
        )
        print(res)

        indicator = lppls_model.compute_indicators(res)

        if self.evaluation:
            lppls_model.plot_confidence_indicators(indicator)

        factor = pd.DataFrame({'date': self.df['date'],
                            'factor': indicator['pos_conf']})

        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)

        return factor

    def augur_0046(self):
        '''
        Turnover rate

        '''

        factor = pd.DataFrame({'date': self.df['date'],
                            'factor': self.df['turnover rate']})

        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)

        return factor

    def augur_0047(self):
        '''
        Volume

        '''

        factor = pd.DataFrame({'date': self.df['date'],
                            'factor': self.df['volume']})

        if self.save:
            tools.save_factor(path=SAVE_PATH, name=sys._getframe().f_code.co_name, factor=factor)

        return factor

def all_main(code, start, end):
    '''
    Calculate factors in the pool: compute all at once.

    Params:
    -------
    code: stocks code for download;
    start & end: time stamp;
    factor_num: the number of the factor to be calculated.

    Return:
    -------
    If 'all': the factors are saved as csv files to path '.factorloader/data/';
    If 'single': return the single calculated factor.

    '''

    Augurs.generate_alphas(start, end, code)


def single_main(code, start, end, factor_num = '0001', save=False, evaluation=False):
    '''
    Calculate factors in the pool: compute a single factor.

    Params:
    -------
    code: stocks code for download;
    start & end: time stamp;
    factor_num: the number of the factor to be calculated.

    Return:
    -------
    If 'all': the factors are saved as csv files to path '.factorloader/data/';
    If 'single': return the single calculated factor.

    '''

    factor = Augurs.generate_alpha_single(alpha_name='augur_'+factor_num, start=start, end=end, code=code, save=save, evaluation=evaluation)

    return factor