import os
import requests
import pandas as pd
import numpy as np
import json
from io import StringIO
import itertools
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics

class inmetData(object):

    def __init__(self):
        self.testPerc = 0.2

    def getDataFrame(self):

        if os.uname().sysname == 'Darwin':
            df = pd.read_csv('../inmet_emp_mensal.csv', index_col=0)
        else:

            # google drive shareable file link
            orig_url = 'https://drive.google.com/file/d/1WQ9NARYCNZxWlQUAf4Zn5TxJcnOGjEn3/view?usp=sharing'

            # get file id
            file_id = orig_url.split('/')[-2]

            # create download url
            dwn_url='https://drive.google.com/uc?export=download&id=' + file_id

            # get raw text inside url
            url = requests.get(dwn_url).text

            # create a buffer
            csv_raw = StringIO(url)

            # read from buffer
            df = pd.read_csv(csv_raw, index_col=0)

        # reset index 
        df.reset_index(drop=True, inplace=True)

        # create sinop and csinop series
        df['date'] = pd.to_datetime(df['date'], format=('%Y-%m-%d'))

        #df.set_index('date', inplace=True)

        return df

    def dataPrep(self, df):

        df['chuva_max_12m'] = df.groupby('cd_estacao')['chuva'].rolling(12).max().reset_index(0,drop=True)

        df['flag_chuva_max_12m'] = np.where((~df['chuva_max_12m'].isna()) & (df['chuva_max_12m'] >= 1),
                                    1,
                                    0)

        fl_year = df[df['flag_chuva_max_12m']==1].groupby('cd_estacao').agg(min_year=('date','min'),
                                                                            max_year=('date','max'),
                                                                            cap=('chuva','max'))

        fl_year['min_year'] = fl_year['min_year'].dt.year+1
        fl_year['max_year'] = fl_year['max_year'].dt.year
        fl_year['cap'] = fl_year['cap']*1.25

        df = pd.merge(df,fl_year,how='left',left_on=df.cd_estacao,right_on=fl_year.index)

        df.drop(columns='key_0', inplace=True)

        df_prep = df[(df['min_year'] <= 2009) & (df['max_year'] >= 2020) & (df.date >= '2009-01-01')].copy()

        return df_prep

    def trainData(self, df):
        testPerc = self.testPerc

        # preparing data for Prophet
        strain = df.rename(columns={'date':'ds','chuva':'y'}).sort_values(['cd_estacao','ds']).reset_index(drop=True)

        # place split date at 80% of avaiable months
        split = len(strain['ds'].unique())*(1-testPerc)
        split_at = strain['ds'][round(split)]
        first_cutoff, last_cutoff = strain['ds'][round(split)-25], strain['ds'][round(split)-13]

        # split and creates floor
        strain = strain[strain['ds'] <= split_at]
        strain['floor'] = 0

        dict_train = {}
        dict_train['df'] = strain
        dict_train['first_cutoff'] = first_cutoff
        dict_train['last_cutoff'] = last_cutoff

        return dict_train

    def prophetTuning(self, df, first_cutoff, last_cutoff):
        cutoffs = pd.date_range(start=first_cutoff, end=last_cutoff, freq='MS')

        param_grid = {  
            #'changepoint_prior_scale': [0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.1, 1.0, 10.0],
            'growth': ['flat']
        }

        # Choose stations

        estacoes = ['A917', 'A211', 'A402', 'A803', 'A727']

        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

        tuning_results = pd.DataFrame([])

        # Use cross validation to evaluate all parameters
        for cd_estacao in estacoes:
            for params in all_params:
                print('\n', cd_estacao, '\n')
                print(params, '\n')
                m = Prophet(**params).fit(df[df.cd_estacao == cd_estacao])  # Fit model with given params
                df_cv = cross_validation(m, cutoffs=cutoffs, horizon='365 days', parallel="processes")
                df_p = performance_metrics(df_cv)
                df_p['params'] = str(params)
                df_p['params_sps'] = params['seasonality_prior_scale']
                df_p['cd_estacao'] = cd_estacao
                tuning_results = tuning_results.append(df_p)
                print('\n')

        return tuning_results

    def prophetTuningOutput(self, df=None):
        if not df:
            df = pd.read_excel(os.path.join('..','tuning_results.xlsx'))

        # select the least std averaged option for the first 99 days
        dfg = df[df.horizon <= 99].groupby(['cd_estacao','params_sps']).agg(mae_std=('mae','std'))
        dfg = dfg.reset_index().sort_values(['cd_estacao','mae_std']).reset_index(drop=True)
        df_output = dfg.drop_duplicates('cd_estacao')[['cd_estacao', 'params_sps']].set_index('cd_estacao')

        # convert to dict
        dict_params = df_output.to_dict()['params_sps']

        # saves the dict

        with open(os.path.join('..','tuning_results.json'), 'w') as f:
            json.dump(dict_params, f)

        return dict_params

    def prophetPredict(self, cd_estacao=None, dict_params=None):
        ## best sinop

        if not dict_params:
            with open(os.path.join('..','tuning_results.json'), 'r') as f:
                dict_params = json.load(f)

        df = self.dataPrep(self.getDataFrame())

        # preparing data for Prophet
        df_run = df.rename(columns={'date':'ds','chuva':'y'}).sort_values(['cd_estacao','ds']).reset_index(drop=True)

        m = Prophet(changepoint_prior_scale = dict_params[cd_estacao],
                    growth='flat').fit(df_run[df_run.cd_estacao == cd_estacao])  # Fit model with given params

        future = m.make_future_dataframe(periods=12, freq='MS')
        forecast = m.predict(future)

        return forecast


    def run(self):
        df = self.getDataFrame()
        df_prep = self.dataPrep(df=df)
        dict_train = self.trainData(df=df_prep)
        tuning_results = self.prophetTuning(**dict_train)
        dfg = self.prophetTuningOutput()
        forecast = self.prophetPredict(cd_estacao='A211')
        forecast.to_excel(os.path.join('..','forecast_A211.xlsx'))
        tuning_results.to_excel(os.path.join('..','tuning_results.xlsx'))


if __name__ == '__main__':
    inmetData().run()