########################################################################################################
library(parallelDist) # https://www.rdocumentation.org/packages/parallelDist/versions/0.1.1/topics/parDist
source("corFuncPtr.R") # 자동으로 패키지 설치 안내문으뜨면 설치하면 된다.
source('df_prepro.R')
library(data.table) # read CSV file
library(dplyr)
library(reshape)
library(forecast) # MA 계산
library(ggplot2) # for vis
library(RColorBrewer) # for vis
library(gridExtra) # for vis
library(bigrquery) # 군집별 특성을 찾을 때 사용
########################################################################################################


setwd('D:\\데이터분석\\초기 6시간')
options(scipen = 999, digits=21)

# 데이터 읽기
 # nid : 유저 아이디
 # rn  : 구분자 번호 (10분단위, 1일단위 등)
 # pt  : rn에 맞는 수치 
data<- fread('SZ_daily_PT_sum_20191025_2.csv')
colnames(data)<- c('nid', 'rn', 'pt')

# Hyper parameter
max_rn <- max(data$rn)
MA <- 3

# 전처리 결과 불러오기 
data<- df_prepro(data,seq_length = max_rn) # source('df_prepro.R')

# COR matrix 작성
 # 유저간 trend 데이터를 담아두는 공간
nid_list<- as.character(unique(data$nid))
COR<- matrix(0, nrow = length(nid_list), ncol = max_rn - MA +1 )
rownames(COR)<- nid_list


# trend 계산
for(i in 1:length(nid_list) ){
  idx<- which(data$nid == nid_list[i])
  
  qq <- forecast::ma(data[idx,'pt'], order = MA)
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
plot(hc1, cex = 0.01, hang = -1, main= 'SZ 2019-10 가입유저 - 가입 후 7일 PT', label = FALSE)
rect.hclust(hc1, k = 3, border = 2:5)


# Cut tree into 3 groups
sub_grp <- cutree(hc1, k = 3)

# Number of members in each cluster
table(sub_grp)



# 시각화
casted_data<- cast(data, nid ~ rn)
pt_matrix<- as.matrix(casted_data[, -1])

# cluster_1
group_num = 1

group_vis<- function(data, group_num){
  idx<- which(sub_grp== group_num)
  idx<- which( as.character(data$nid) %in% names(sub_grp[idx]) )
  
  temp<- data[idx,]
  temp$nid<- as.factor(temp$nid)
  
  p1<-ggplot(temp,  aes(rn, nid)) +
    geom_tile(aes(fill = pt)) +
    scale_fill_gradientn(colors = rev(brewer.pal(11, "RdBu"))) +
    scale_y_discrete(limits = rev(unique(temp$nid))) +
    ggtitle(paste('User Group',group_num, "by 1 DAY")) +
    theme(panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.border = element_blank(),
          panel.background = element_blank(),
          axis.text.x = element_text(angle = 90),
          axis.ticks.y=element_blank(),
          axis.text.y=element_blank())
  
  
  idx<- which(sub_grp==group_num)
  group_1<- data.frame(mean_pt =colMeans(pt_matrix[idx,]), group = 1:length(colMeans(pt_matrix[idx,])))
  
  p2<- ggplot(group_1, aes(group, mean_pt))+
    geom_line(size=2)+ #ylim(0,350)+
    ggtitle("User Group Mean Play Time")+
    theme(axis.ticks.x=element_blank(),
          
          axis.title.x=element_blank())
  
  
  p3<- arrangeGrob(p2, p1, nrow=2, top= paste('Group', group_num))
  return(p3)  
}

vis_group_1<- group_vis(data, group_num = 1)
grid.arrange(vis_group_1)

vis_group_2<- group_vis(data, group_num = 2)
grid.arrange(vis_group_2)

vis_group_3<- group_vis(data, group_num = 3)
grid.arrange(vis_group_3)





# 유저 특성 찾기
 # Provide authentication through the JSON service account key
path="D:/데이터분석/SZ/lgsz-0718-5728f5afdf4f.json"
bq_auth(path)

# Store the project id
projectid="lgsz-0718"

# Set your query
group_num = 1
grp_nid<- names( sub_grp[ which(sub_grp == group_num)] )


sql <- paste0(" 
              with sample as (
              SELECT nid,  max_pl
              FROM sz_dw.f_user_map
              where  date_diff_reg = 0
              and nid in ('",  paste( grp_nid, collapse = "','" ), "')"
              , ")
              
              select 1.0 * sum(lv) / count(distinct nid)
              from sample as A
              LEFT JOIN sz_dw.dim_hero_lv as B
              ON A.max_pl = B.lv_hero
              "
)


# Run the query and store the data in a dataframe
tb <- bq_project_query(query=sql,x=projectid) 
df <- bq_table_download(tb)
print(df)



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