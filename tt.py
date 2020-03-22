# pandas-gbq
from google.oauth2 import service_account # google-oauth
import pandas as pd
import os
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import curve_fit
import numpy as np

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth',200)

# 이삭 빅쿼리 접근 키
json_dir = './bigquery'
json_path = os.path.join(json_dir, 'EH' + '.json')
credentials = service_account.Credentials.from_service_account_file(json_path)

def sqldf(qry):
    output = pd.read_gbq(query=qry,
                         project_id='lgeh-251102',
                         credentials=credentials,
                         dialect='standard')
    return output


query = '''
with sample as(
  select A.reg_datekey, A.datekey, reg_uu, date_diff(A.datekey, A.reg_datekey, DAY) as reg_diff, DAU
  , coalesce(DAU , 0)/ reg_uu as retention
  , coalesce(daily_revenue , 0)/ reg_uu as ARPU
  from(
    select reg_datekey, datekey, count(distinct nid) as DAU, sum(daily_revenue) as daily_revenue
    from eh_dw.f_user_map
    where login_flag = 1
    and datekey>= '2019-11-21' and datekey<= '2019-12-31'
    and reg_datekey = '2019-11-24'
    group by reg_datekey, datekey
  )as A
  INNER JOIN(
    -- 인원수 확인
    select reg_datekey, count(distinct nid) as reg_uu
    from eh_dw.f_user_map
    where nru = 1
    and datekey>= '2019-11-21' and datekey<= '2019-12-31'
    and reg_datekey = '2019-11-24'
    group by reg_datekey
  )as B
  ON A.reg_datekey = B.reg_datekey
)
, sample2 as (
  select reg_datekey, datekey, reg_uu, reg_diff, DAU
  , retention, ARPU
  , sum( retention * ARPU ) over(partition by reg_datekey order by datekey rows between unbounded preceding and current row)as LTV
  from sample
)

select A.datekey, A.reg_datekey, DAU*LTV as daily_revenue_hat, B.daily_revenue
,B.daily_revenue - DAU*LTV  as error
, retention
, ARPU,DAU, reg_diff
from (
  select *
  from sample2
  where reg_diff <= 40
)as A
INNER JOIN (
  select reg_datekey,datekey, sum(daily_revenue) as daily_revenue
  from eh_dw.f_user_map
  where datekey>= '2019-11-21' and datekey<= '2019-12-31'
  and reg_datekey = '2019-11-24'
  group by reg_datekey,datekey
)as B
ON A.datekey = B.datekey and A.reg_datekey = B.reg_datekey
order by reg_datekey, datekey'''
df = sqldf(query)
df = df.fillna(0)
## 전처리 --> 전체 유저들의 매출액
df.head(3)
df.reg_datekey = df.reg_datekey.dt.strftime('%Y-%m-%d')

# 분석에 사용할 데이터 df_s( df_sample) 로 정의
df_s = df[df.reg_diff <= 8]


# 전처리
N = df_s.reg_diff.max() +1 # 데이터 갯수 최댓값
M = df.shape[0] # 예측단위
df_N = df_s.groupby('reg_datekey')['reg_diff'].max() # 가입일별 데이터 갯수
df_N = pd.DataFrame(df_N.values)
df_N.columns = colnames


x_bar = df_s.groupby('reg_datekey')['retention'].mean() # 평균
sd = df_s.groupby('reg_datekey')['retention'].var() # 분산

alpha_hat = x_bar* ( (x_bar * (1-x_bar)) / sd -1) # alpha 추정값
beta_hat = (1-x_bar)* ( (x_bar * (1-x_bar)) / sd -1) # beta 추정값



# S(T=t) 계산
# 'S(T=t)': beta_hat / (alpha_hat + beta_hat)
tt = pd.DataFrame( beta_hat / (alpha_hat + beta_hat))#.reset_index()
tt = tt.T.reset_index(drop =True)

survival_df_s = pd.DataFrame(np.repeat(tt.values, M, axis=0))
survival_df_s.columns = tt.columns
survival_df_s = survival_df_s.reset_index()
survival_df_s = survival_df_s.rename(columns = {'index':'t'})
del survival_df_s.columns.name
print(survival_df_s)

colnames = list(survival_df_s.columns)
colnames.remove('t')
ncol = len(colnames)
# ratio_t 계산
ratio_df_s = pd.DataFrame(np.repeat(np.zeros((1, ncol)), M, axis=0))
ratio_df_s.columns = colnames

for col in colnames :
    for t in range(2, M):
        ratio_df_s.loc[t, col] = (beta_hat[col] + t - 1) / (alpha_hat[col] + beta_hat[col] + t -1 )

for col in colnames :
    for i in range(2, M):
        survival_df_s.loc[i ,col] = survival_df_s.iloc[i-1,][col] * ratio_df_s.iloc[i,][col]

# survival_df_s: if t = 0 : retention is 1.
survival_df_s.loc[survival_df_s.t == 0, colnames ] = 1


#DAU 계산 - df에서 가입일별 DAU 가져오기
DAU = pd.DataFrame(np.zeros(shape = (df.shape[0], ncol)))
DAU.columns = colnames
for col in colnames:
    idx = df.reg_datekey == col
    DAU[col] = df.loc[idx, 'DAU']



#DAU_hat 계산하기
DAU_hat = pd.DataFrame(np.zeros(shape = (df.shape[0], ncol)))
DAU_hat.columns = colnames

for col in colnames:
    idx = df_s.reg_datekey == col
    DAU_hat[col] = df_s.loc[idx, 'DAU']

    idx2 = np.where(DAU_hat[col].isnull())[0][0] # 첫 True 위치

    for j in range(df.shape[0]):
        if j > idx2-1 :
            DAU_hat.loc[j, col] = DAU_hat.loc[0, col] * survival_df_s.loc[j-1, col]

        else:
            DAU_hat.loc[j, col] = DAU.loc[j, col]

# MA(7) ARPU계산
ARPU_hat = pd.DataFrame(np.zeros(shape = (M, ncol)))
ARPU_hat.columns = colnames

# 예측의 결과값 저장
pred = pd.DataFrame(np.zeros(shape = (M, ncol)))
pred.columns = colnames
# pred에 첫 7일은 넣어주기. --> 이미 알고 있는 내용


# 실제 데이터 저장
real = pred.copy(deep = True)

for col in colnames:
    idx = df_s.reg_datekey == col
    ARPU_hat[col] = df_s.loc[idx, 'ARPU'].rolling(window = 7, min_periods= 1).mean()

    idx2 = np.where(ARPU_hat[col].isnull())[0][0] # 첫 True 위치
    max_n = df_N[col].values

    for j in range(df.shape[0]):
        if j <= max_n :
            pred.loc[j, col] = ARPU_hat.loc[j, col] * DAU_hat.loc[j, col] * survival_df_s.loc[:j,col].sum()  # 실제값으로 바꾸기!!!
        elif 6 <= j <= max_n :
            pred.loc[j, col] = ARPU_hat.loc[j, col] * DAU_hat.loc[j, col] * survival_df_s.loc[:j, col].sum()
        else:
            pred.loc[j, col] = ARPU_hat.loc[idx2 - 1, col] * DAU_hat.loc[j, col] * survival_df_s.loc[:j, col].sum()


            ### 실제 매출액과 비교
    real[col] = df.loc[df.reg_datekey == col, 'daily_revenue']



f, ax = plt.subplots(1, 1)
plt.plot(real[col])
plt.plot(pred[col])
plt.show()


### error 계산





f, ax = plt.subplots(1, 1)
plt.plot(survival_df_s[col])
plt.plot(df['retention'])
plt.show()

f, ax = plt.subplots(1, 1)
plt.plot(DAU_hat[col])
plt.plot(df['DAU'])
plt.show()




# sBG =survival_df_s['S(T=t)']
# sBG = sBG.values


# Gaussian model을 리턴합니다.
def func(x, b, c, d):
    return d / (b * x + c)

# 마찬가지로 0~10까지 100개 구간으로 나눈 x를 가지고
x = np.linspace(0, N, N)
y = func(x = x, b = 5, c =2, d =3) # 답인 y들과
yn = y + 0.2*np.random.normal(size=len(x)) # noise가 낀 y값들을 만듭니다.

## 그런 후에 curve_fit을 하고 best fit model의 결과를 출력합니다.
popt, pcov = curve_fit(func, x, yn)
print(popt)
print(pcov)



plt.scatter(x, yn, marker='.')
plt.plot(x, y, linewidth=2, c = 'b')
plt.plot(x, func(x, * popt), color='red', linewidth=2)
plt.plot(x, sBG, color='black', linewidth=2)
plt.legend(['Original', 'Best Fit', 'sGB'], loc=2)
plt.show()
