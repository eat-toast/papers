# pandas-gbq
from google.oauth2 import service_account # google-oauth
import pandas as pd
import os
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from sklearn.metrics import mean_squared_error

# custom Python files
from revenew_predict.utils.load_query_functions import load_query

pd.options.display.float_format = '{:.2f}'.format

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth',200)


# 이삭 빅쿼리 접근 키
json_dir = './bigquery'
project_name = 'EH'
project_set = {'RC':'lgrc-271504', 'EH':'lgeh-251102'}


json_path = os.path.join(os.getcwd(),'bigquery', project_name + '.json')
credentials = service_account.Credentials.from_service_account_file(json_path)

def sqldf(qry):
    output = pd.read_gbq(query=qry,
                         project_id=project_set[project_name],
                         credentials=credentials,
                         dialect='standard')
    return output


# 예측할 기간
multiple_predict = True
# model을 생성할 train 기간을 설정한다.
    # ex) RC론칭일부터 적당한 일자까지 array를 생성한다.


reg_date_range = [d.strftime('%Y-%m-%d') for d in pd.date_range('2020-05-28','2020-06-11')]

predict_min_date = min(reg_date_range)
predict_max_date = '2020-07-15'
predict_date_range = [d.strftime('%Y-%m-%d') for d in pd.date_range(predict_min_date,predict_max_date)]


query = load_query(multiple_predict, project_name, reg_datekey_str=reg_date_range, predict_min_date = predict_min_date, predict_max_date = predict_max_date)

# 데이터 불러오기
df = sqldf(query)
df = df.fillna(0)


def make_sample_data(raw_data, use_data_cnt, max_data_cnt = 30):
    # ex 론칭 7일차 데이터 이용

    # reg_diff가 use_data_cnt 보다 큰 가입 일자만 가져오기
    raw_data = raw_data.loc[raw_data.reg_diff < max_data_cnt]
    reg_date_cond = raw_data.groupby('reg_datekey')['reg_diff'].max().reset_index()
    df_s_reg_date_cond = reg_date_cond.loc[reg_date_cond.reg_diff >= use_data_cnt, 'reg_datekey'].dt.strftime('%Y-%m-%d')

    df_s = raw_data.loc[raw_data.reg_datekey.isin(df_s_reg_date_cond)]


    # 전처리
    df_s.reg_datekey = df_s.reg_datekey.dt.strftime('%Y-%m-%d')
    df_s.datekey = df_s.datekey.dt.strftime('%Y-%m-%d')

    return df_s


def ff2(df, predict_date_range, reg_date_range, window_days=7):
    '''
    DAU, Retention등, 예측할 일자까지 저장하기
    '''
    '''
    :param df: from Bigquery 
    :param window_days: moving averaging - ARPU window size
    :return:
      real: 실제 매출액
      pred_ratio: ratio function 예측 매출액
    '''
    # 론칭 후 6일 (0 ~ 6일 7개 데이터)만 이용하기
    use_data_cnt = 6

    # 분석에 사용할 데이터 df_s(df_sample) 로 정의
    df_s = make_sample_data(raw_data = df, use_data_cnt= use_data_cnt)
    colnames = df_s.reg_datekey.unique()  # 가입일 정보
    ncol = len(colnames) # 컬럼 수

    N = df_s.reg_diff.max() + 1  # 데이터 갯수 최댓값
    M = 30  # df.groupby('reg_datekey')['reg_diff'].max().min()
    NAN_start = df_s.datekey.max()

    df_N = df_s.groupby('reg_datekey')['reg_diff'].max()  # 가입일별 데이터 갯수
    df_N = pd.DataFrame(df_N.values).T
    df_N.columns = colnames

    ######################## 유리 함수 curve_fit ########################

    # Ratio Function 정의
    def func(x, b, c):
        return 1 / (b * x + c)

    # 유리함수 fitting에 사용 될 retention
    ratio_func = pd.DataFrame(np.nan, index = predict_date_range, columns = reg_date_range)
    ratio_func_fit = pd.DataFrame(np.nan, index=predict_date_range, columns=reg_date_range)

    # 가입일자에 맞게 retention 저장
    for col in colnames:
        temp = df_s.loc[df_s.reg_datekey == col, ['datekey','retention']]
        temp.set_index('datekey', inplace = True)
        ratio_func.loc[col:, col] = temp.retention

    # curve fitting
    for col in colnames:
        not_null_idx = ratio_func.loc[col:, col].notnull()
        predict_date = not_null_idx.shape[0]

        yn = ratio_func.loc[col:, col][not_null_idx].values
        x = np.array([x for x in range(not_null_idx.sum())])

        x2 = np.arange(0, predict_date) # 예측하고자 하는 일자

        ## curve_fit을 하고 best fit model의 결과를 출력
        popt, pcov = curve_fit(func, x, yn)
        ratio_func_fit.loc[col:, col] = func(x2, * popt)

    ######################## DAU ########################
    # DAU_hat 계산하기
    DAU_hat_ratio = pd.DataFrame(np.nan, index = predict_date_range, columns = reg_date_range)

    for col in colnames:
        idx = df_s.reg_datekey == col
        DAU1 = df_s.loc[idx, ['datekey','DAU']].reset_index(drop=True)  # 실제 DAU값 가져오기
        DAU1.set_index('datekey', inplace=True)
        index_hubo = [d.strftime('%Y-%m-%d') for d in pd.date_range(DAU1.index[-1], max(predict_date_range))][1:]  # 0번째는 ARPU_MA가 가지고 있어서 제외
        DAU2 = pd.DataFrame(np.nan, index=index_hubo, columns=['DAU'])

        DAU = pd.concat([DAU1, DAU2])
        DAU_hat_ratio[col] = DAU

        # for j in range(len(predict_date_range)):
        #     # if j <= df_N[col].values :  # N
        #     #     DAU_hat_ratio.loc[predict_date_range[j], col] = DAU.loc[predict_date_range[j]].values
        #
        #     if j > df_N[col].values and j + 1 < len(predict_date_range):  # N
        #         DAU_hat_ratio.loc[predict_date_range[j], col] = DAU_hat_ratio.loc[col, col] * ratio_func_fit.loc[
        #             predict_date_range[j + 1], col]

        for col_idx in predict_date_range[predict_date_range.index(NAN_start)+1:]:
            DAU_hat_ratio.loc[col_idx, col] = DAU_hat_ratio.loc[col, col] * ratio_func_fit.loc[col_idx, col]

    ######################## MA(7) ARPU계산 ########################
    ARPU_hat = pd.DataFrame(np.nan, index=predict_date_range, columns=reg_date_range)

    for col in colnames:
        idx = df_s.reg_datekey == col
        ARPU_MA = df_s.loc[idx, ['datekey','ARPU']].set_index(keys = 'datekey').rolling(window=window_days, min_periods=window_days).mean()
        index_hubo = [d.strftime('%Y-%m-%d') for d in pd.date_range(ARPU_MA.index[-1], max(predict_date_range))][1:] # 0번째는 ARPU_MA가 가지고 있어서 제외
        ARPU_MA2 = pd.DataFrame(np.nan, index =index_hubo, columns = ['ARPU'])

        ARPU_MA = pd.concat([ARPU_MA, ARPU_MA2])


        ARPU_hat.loc[:, col] = ARPU_MA

        # 이동평균 이후의 NA는 마지막값으로 채운다.
        idx = ARPU_hat.loc[:, col].notnull()
        last_ma = ARPU_hat.loc[idx, col].iloc[-1]
        pp = ARPU_hat.loc[idx, col].index[-1]

        ARPU_hat.loc[pp:, col] = ARPU_hat.loc[pp:, col].fillna(last_ma)

    # 예측의 결과값 저장
    pred_ratio = pd.DataFrame(np.nan, index=predict_date_range, columns=reg_date_range)
    # 실제 데이터 저장
    real = pred_ratio.copy(deep=True)


    for col in colnames:

        # 기존에 이미 알고 있는 값
        temp_real_s = df_s.loc[df_s.reg_datekey == col, ['datekey', 'daily_revenue']].reset_index(drop=True)  # 실제 DAU값 가져오기
        temp_real_s.set_index('datekey', inplace = True)
        pred_ratio.loc[:, col] = temp_real_s

        index_hubo = [d.strftime('%Y-%m-%d') for d in pd.date_range(temp_real_s.index[-1], max(predict_date_range))][
                     1:]  # 0번째는 ARPU_MA가 가지고 있어서 제외

        for j in index_hubo:
            pred_ratio.loc[j, col] = DAU_hat_ratio.loc[j, col] * ARPU_hat.loc[j , col] #* (0.99 ** np.where(DAU_hat_ratio.index==j)[0][0])

        ### 실제 매출액과 비교
        temp_real = df.loc[df.reg_datekey == col, ['datekey','daily_revenue']].reset_index(drop=True)
        temp_real.datekey= temp_real.datekey.dt.strftime('%Y-%m-%d')
        temp_real.set_index('datekey', inplace = True)

        real.loc[:,col] = temp_real
        real = real.fillna(0)



    # 학습기간 7일이 안된 가입일자는 가입일 최근 7일 평균값 넣기
    last_7_reg_datekey = df_N.columns[-7:]
    post_fix = list(set(reg_date_range)-set(df_N.columns)) # 변수명... 몰로 바꾸지..
    if len(post_fix) > 0 :
        for start_row in post_fix:
            pred_ratio.loc[start_row:, last_7_reg_datekey].mean(axis=1)
            pred_ratio.loc[start_row:, start_row] = pred_ratio.loc[start_row:, last_7_reg_datekey].mean(axis=1)

        for start_row in post_fix:
            idx = df.reg_datekey == start_row
            tt = df.loc[idx,['datekey', 'daily_revenue']]
            tt.datekey = tt.datekey.dt.strftime('%Y-%m-%d')
            tt.set_index('datekey', inplace=True)
            real.loc[:, start_row] = tt

    return pred_ratio, real


pred_ratio, real = ff2(df, predict_date_range, reg_date_range, window_days=7)



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
