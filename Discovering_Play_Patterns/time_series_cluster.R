setwd('D:\\git_R\\papers\\Discovering_Play_Patterns')
options(scipen = 999, digits=21)
########################################################################################################
library(parallelDist) # https://www.rdocumentation.org/packages/parallelDist/versions/0.1.1/topics/parDist
source("corFuncPtr.R") # 자동으로 패키지 설치 안내문으뜨면 설치하면 된다.
source('df_prepro.R')
source('group_vis.R')
library(data.table) # read CSV file
library(dplyr)
library(reshape)
library(forecast) # MA 계산
library(ggplot2) # for vis
library(RColorBrewer) # for vis
library(gridExtra) # for vis
library(bigrquery) # 군집별 특성을 찾을 때 사용
########################################################################################################


# 데이터 읽기
 # nid : 유저 아이디
 # rn  : 구분자 번호 (10분단위, 1일단위 등)
 # pt  : rn에 맞는 수치 
data<- fread('SZ_daily_PT_sum_201909.csv', integer64 = 'numeric')
colnames(data)<- c('nid', 'rn', 'pt')

# Hyper parameter
max_rn <- max(data$rn)
MA <- 3

# 전처리 결과 불러오기 
data<- df_prepro(data,seq_length = max_rn) # source('df_prepro.R')


# 전처리 단계 추가 --> rn 2이상 sum(pt) = 0 이면 data에서 제외
 # 보통 30 ~ 50% 유저들이 걸러진다.
nid_pt_sum<-data %>% filter(rn > 1) %>% group_by(nid) %>% summarise(sum_pt = sum(pt))
nid_pt_sum<- nid_pt_sum %>% filter(sum_pt==0)
data<- data %>% filter( !(nid %in% nid_pt_sum$nid ))


# COR matrix 작성
 # 유저간 trend 데이터를 담아두는 공간
nid_list<- as.character(unique(data$nid))
COR<- matrix(0, nrow = length(nid_list), ncol = max_rn - MA +1 )
rownames(COR)<- nid_list


# trend 계산
 # nid(유저)별 모든 seq데이터를 생성해 놨기 때문에 아래와 같이 수행 가능
 # max_rn 의 크기만큼 nid별 인덱스를 생성
start_list<- seq(1, nrow(data), by = max_rn)
end_list<- seq(1+max_rn-1, nrow(data)+max_rn-1, by = max_rn)

temp_list<- vector(mode ='list', length = length(nid_list))
for(i in 1:length(nid_list)){
  temp_list[[i]]<- start_list[i] : end_list[i]  
}
for(i in 1:length(nid_list)){
  qq <- forecast::ma(data[temp_list[[i]],'pt'], order = MA)
  qq <- qq[!is.na(qq)]  
  COR[i,]<-qq
}




# COR + trend 구하기
d<- parDist(COR, method="custom", func = corFuncPtr)
d[is.na(d)]<-0

# 군집
  # Hierarchical clustering using Ward Linkage
hc1 <- fastcluster::hclust(d, method = "ward.D" )

# Plot the obtained dendrogram
 # 데이터가 많으면 굳이 안그릴 것을 추천
# plot(hc1, cex = 0.01, hang = -1, main= 'SZ 2019-10 가입유저 - 가입 후 7일 PT', label = FALSE)
# rect.hclust(hc1, k = 6, border = 2:8)


# Cut tree into 3 groups
sub_grp <- cutree(hc1, k = 6)

# Number of members in each cluster
table(sub_grp)



# 시각화
casted_data<- cast(data, nid ~ rn)
pt_matrix<- as.matrix(casted_data[, -1])


vis_group_1<- group_vis(data, group_num = 1)
#grid.arrange(vis_group_1)

vis_group_2<- group_vis(data, group_num = 2)
#grid.arrange(vis_group_2)

vis_group_3<- group_vis(data, group_num = 3)
#grid.arrange(vis_group_3)

vis_group_4<- group_vis(data, group_num = 4)
#grid.arrange(vis_group_4)

vis_group_5<- group_vis(data, group_num = 5)
#grid.arrange(vis_group_5)

vis_group_6<- group_vis(data, group_num = 6)
#grid.arrange(vis_group_6)

grid.arrange(vis_group_1, vis_group_2, vis_group_3, vis_group_4, vis_group_5, vis_group_6)





# 유저 특성 찾기
 # Provide authentication through the JSON service account key
path="D:/데이터분석/SZ/lgsz-0718-5728f5afdf4f.json"
bq_auth(path)

# Store the project id
projectid="lgsz-0718"

# Set your query
level_sql <- paste0(" 
              with sample as (
              SELECT nid,  max_pl
              FROM sz_dw.f_user_map
              where  date_diff_reg = 0
              and nid in ('",  paste( grp_nid, collapse = "','" ), "')"
              , ")
              
              select 1.0 * sum(lv) / count(distinct nid) as avg_pl
              from sample as A
              LEFT JOIN sz_dw.dim_hero_lv as B
              ON A.max_pl = B.lv_hero
              "
)

bigquery_sql<- function(sql, projectid, num_group){
  result<- data.frame()
  for( num in seq_len(num_group) ){
    grp_nid<- names( sub_grp[ which(sub_grp == num)] )
    
    # Run the query and store the data in a dataframe
    tb <- bq_project_query(query=sql,x=projectid) 
    df <- bq_table_download(tb)
    result<- rbind(result, df)
  }
  
  return(result)
}







#### 그룹별 결제금액
sql <- paste0(" 
              with sample as (
              SELECT nid, sum(daily_revenue) as daily_revenue
              FROM sz_dw.f_user_map
              where  date_diff_reg < 7 and daily_revenue > 0 
              and nid in ('",  paste( grp_nid, collapse = "','" ), "')
              group by nid"
              
              , ")
              
              select 1.0 * sum(daily_revenue) / count(distinct nid) as avg_revenue, count(distinct nid) as pu
              from sample 
              "
)

# Run the query and store the data in a dataframe
tb <- bq_project_query(query=sql,x=projectid) 
df <- bq_table_download(tb)
print(df)