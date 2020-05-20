# pandas-gbq
from google.oauth2 import service_account # google-oauth
import pandas as pd
import os
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import datetime

# custom Python files
from revenew_predict.utils.load_query_functions import load_query

pd.options.display.float_format = '{:.2f}'.format

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth',200)


# 이삭 빅쿼리 접근 키
json_dir = './bigquery'
project_name = 'RC'
project_set = {'RC':'lgrc-271504', 'EH':'lgeh-251102'}


json_path = os.path.join(os.getcwd(),'bigquery', project_name + '.json')
credentials = service_account.Credentials.from_service_account_file(json_path)


def make_sample_data(df, use_data_cnt):
    df_s = df[ df.reg_diff <= use_data_cnt ] # ex 론칭 7일차 데이터 이용
    idx = df.reg_diff <= use_data_cnt

    temp = df.loc[idx,].groupby('reg_datekey')['reg_diff'].max().reset_index()
    temp = temp.loc[temp.reg_diff >= (use_data_cnt), 'reg_datekey']

    df_s = df_s.loc[df_s.reg_datekey.isin(temp)]

    return df_s


def sqldf(qry):
    output = pd.read_gbq(query=qry,
                         project_id=project_set[project_name],
                         credentials=credentials,
                         dialect='standard')
    return output


# 예측할 기간
M = 30
multiple_predict = True
# model을 생성할 train 기간을 설정한다.
    # ex) RC론칭일부터 적당한 일자까지 array를 생성한다.
reg_date_range = pd.date_range(start = '2020-04-29', end = '2020-05-15' )
reg_datekey_30 = max(reg_date_range) + datetime.timedelta(days=M)

predict_date_range = pd.date_range(start = min(reg_date_range), end =  reg_datekey_30)


real_df = pd.DataFrame(0,  index = predict_date_range, columns =predict_date_range) # 실제 매출액을 담아둘 공간
pred_ratio_df = pd.DataFrame(0,  index = predict_date_range, columns =predict_date_range) # 예상 매출액을 담아둘 공간


def ff(df, window_days = 7):
    '''
    :param df: from Bigquery 
    :param window_days: moving averaging - ARPU window size
    :return:
      real: 실제 매출액
      pred_ratio: ratio function 예측 매출액
    '''
    # 론칭 후 6일 (0 ~ 6일 7개 데이터)만 이용하기
    use_data_cnt = 6

    # 분석에 사용할 데이터 df_s( df_sample) 로 정의
    df_s = make_sample_data(df, use_data_cnt)
    colnames = df_s.reg_datekey.unique()  # 가입일 정보
    ncol = len(colnames) # 컬럼 수

    # 전처리
    N = df_s.reg_diff.max() + 1  # 데이터 갯수 최댓값
    M = 30  # df.groupby('reg_datekey')['reg_diff'].max().min()
    df_N = df_s.groupby('reg_datekey')['reg_diff'].max()  # 가입일별 데이터 갯수
    df_N = pd.DataFrame(df_N.values).T
    df_N.columns = colnames

    ######################## 유리 함수 curve_fit ########################

    # Ratio Function 정의
    def func(x, b, c):
        return 1 / (b * x + c)

    # 유리함수 fitting에 사용 될 retentnion
    ratio_func = pd.DataFrame(np.repeat(np.zeros((1, ncol)), N, axis=0), columns = colnames)
    ratio_func_fit = pd.DataFrame(np.repeat(np.zeros((1, ncol)), M, axis=0), columns = colnames)

    # 가입일자에 맞게 retention 저장
    for col in colnames:
        ratio_func[col] = df_s.loc[df_s.reg_datekey == col, 'retention'].values

    # curve fitting
    for col in colnames:
        x = ratio_func[col].index.values
        yn = np.ravel(ratio_func[col].values)
        x2 = np.arange(0, M) # 예측하고자 하는 일자

        ## curve_fit을 하고 best fit model의 결과를 출력
        popt, pcov = curve_fit(func, x, yn)
        ratio_func_fit[col] = func(x2, * popt)
    
    
    ######################## DAU ########################
    # DAU_hat 계산하기
    DAU_hat_ratio = pd.DataFrame(np.zeros(shape=(M, ncol)), columns = colnames)

    for col in colnames:
        idx = df_s.reg_datekey == col
        DAU_hat_ratio[col] = df_s.loc[idx, 'DAU'].reset_index(drop=True)

        for j in range(M):
            if j >= N:
                DAU_hat_ratio.loc[j, col] = DAU_hat_ratio.loc[0, col] * ratio_func_fit.loc[j, col]

    ######################## MA(7) ARPU계산 ########################
    ARPU_hat = pd.DataFrame(np.nan, index=range(M), columns=colnames)

    for col in colnames:
        idx = df_s.reg_datekey == col
        ARPU_MA = df_s.loc[idx, 'ARPU'].rolling(window = window_days, min_periods= window_days).mean().reset_index(drop=True)

        not_null_rownumber = ARPU_MA.shape[0] - 1

        ARPU_hat.loc[:not_null_rownumber, col] = ARPU_MA
        ARPU_hat.loc[:not_null_rownumber, col] = ARPU_hat.loc[:not_null_rownumber, col].fillna(0)

        # 이동평균 이후의 NA는 마지막값으로 채운다.
        idx = ARPU_hat.loc[:, col].notnull()
        last_ma = ARPU_hat.loc[idx, col].iloc[-1]

        ARPU_hat.loc[~idx, col] = ARPU_hat.loc[~idx, col].fillna(last_ma)

    # 예측의 결과값 저장
    pred_ratio = pd.DataFrame(np.zeros(shape=(M, ncol)), columns = colnames)
    # 실제 데이터 저장
    real = pred_ratio.copy(deep=True)


    for col in colnames:
        idx = DAU_hat_ratio[col] == 0
        min_n = idx.sum()
        max_n = df_N[col].values[0]

        # 기존에 이미 알고 있는 값
        temp_real_s = df_s.loc[df_s.reg_datekey == col, 'daily_revenue'].reset_index(drop=True)
        pred_ratio.loc[min_n:(min_n + max_n), col] = temp_real_s.values

        for j in range(min_n, M):
            if min_n + max_n < j:
                pred_ratio.loc[j, col] = ARPU_hat.loc[j - 1, col] * DAU_hat_ratio.loc[j, col]
        ### 실제 매출액과 비교
        temp_real = df.loc[df.reg_datekey == col, 'daily_revenue'].reset_index(drop=True)
        real[col] = temp_real[: M - min_n]
        real = real.fillna(0)
    return real, pred_ratio


if multiple_predict:

    query = load_query(multiple_predict, project_name, reg_datekey_str = reg_date_range )

    df = sqldf(query)
    df = df.fillna(0)
    ## 전처리 --> 전체 유저들의 매출액
    print(df.head(3))

    df.reg_datekey = df.reg_datekey.dt.strftime('%Y-%m-%d')
    df.datekey = df.datekey.dt.strftime('%Y-%m-%d')

    real, pred_ratio = ff(df)
    ### 결과 저장
    real_df.loc[reg_datekey_str:reg_datekey_30, reg_datekey_str] = real.values
    pred_ratio_df.loc[reg_datekey_str:reg_datekey_30, reg_datekey_str] = pred_ratio.values



if multiple_predict:
    for reg_datekey in reg_datekey_range:
        reg_datekey_str = reg_datekey.strftime('%Y-%m-%d')
        reg_datekey_30 = (reg_datekey + datetime.timedelta(days=29)).strftime('%Y-%m-%d')

        query = load_query(project_name, reg_datekey_str)

        df = sqldf(query)
        df = df.fillna(0)
        ## 전처리 --> 전체 유저들의 매출액
        print(df.head(3))

        df.reg_datekey = df.reg_datekey.dt.strftime('%Y-%m-%d')
        df.datekey = df.datekey.dt.strftime('%Y-%m-%d')

        real, pred_ratio = ff(df)
        ### 결과 저장
        real_df.loc[reg_datekey_str:reg_datekey_30, reg_datekey_str] = real.values
        pred_ratio_df.loc[reg_datekey_str:reg_datekey_30, reg_datekey_str] = pred_ratio.values


np.abs(real_df - pred_ratio_df).sum(axis = 1)


real_df.sum(axis = 1).plot(label = 'real')
pred_ratio_df.sum(axis = 1).plot(label = 'ratio function predict')
plt.axvline( reg_datekey_range[0]+ datetime.timedelta(days=6) , color = 'r')
plt.legend()
plt.tight_layout()
plt.show()

idx = pred_ratio_df.index.day != 1
real_df.loc[idx].sum(axis = 1).plot()
pred_ratio_df.loc[idx].sum(axis = 1).plot()