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
json_dir = 'D:\\bigquery'
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
    and datekey>= '2019-11-21' and datekey<= '2019-11-28'
    and reg_datekey <= '2019-11-22'
    group by reg_datekey, datekey
  )as A
  INNER JOIN(
    -- 인원수 확인
    select reg_datekey, count(distinct nid) as reg_uu
    from eh_dw.f_user_map
    where nru = 1
    and datekey>= '2019-11-21' and datekey<= '2019-11-28'
    and reg_datekey <= '2019-11-22'
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
  where datekey>= '2019-11-21' and datekey<= '2019-11-28'
  and reg_datekey <= '2019-11-22'
  group by reg_datekey,datekey
)as B
ON A.datekey = B.datekey and A.reg_datekey = B.reg_datekey
order by reg_datekey, datekey'''
df = sqldf(query)
df = df.fillna(0)
## 전처리 --> 전체 유저들의 매출액
df.head(3)
df.reg_datekey = df.reg_datekey.dt.strftime('%Y-%m-%d')

# 전처리
N = df.reg_diff.max() +1 # 데이터 갯수 최댓값
M = 30 # 예측단위

x_bar = df.groupby('reg_datekey')['retention'].mean() # 평균
sd = df.groupby('reg_datekey')['retention'].var() # 분산

alpha_hat = x_bar* ( (x_bar * (1-x_bar)) / sd -1) # alpha 추정값
beta_hat = (1-x_bar)* ( (x_bar * (1-x_bar)) / sd -1) # beta 추정값


# MA(7) ARPU계산


# S(T=t) 계산
# 'S(T=t)': beta_hat / (alpha_hat + beta_hat)
tt = pd.DataFrame( beta_hat / (alpha_hat + beta_hat))#.reset_index()
tt = tt.T.reset_index(drop =True)

survival_df = pd.DataFrame(np.repeat(tt.values, M, axis=0))
survival_df.columns = tt.columns
survival_df = survival_df.reset_index()
survival_df = survival_df.rename(columns = {'index':'t'})
del survival_df.columns.name
print(survival_df)

colnames = list(survival_df.columns)
colnames.remove('t')

# ratio_t 계산
ratio_df = pd.DataFrame(np.repeat(np.zeros((1, 2)), M, axis=0))
ratio_df.columns = colnames

for col in colnames :
    for t in range(2, M):
        ratio_df.loc[t, col] = (beta_hat[col] + t - 1) / (alpha_hat[col] + beta_hat[col] + t -1 )

for col in colnames :
    for i in range(2, M):
        survival_df.loc[i ,col] = survival_df.iloc[i-1,][col] * ratio_df.iloc[i,][col]

# survival_df: t > 0
survival_df = survival_df[survival_df.t > 0]




















# sBG =survival_df['S(T=t)']
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
