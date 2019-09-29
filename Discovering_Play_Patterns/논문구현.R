########################################################################################################
library(data.table)
library(dplyr)

setwd('C:\\Users\\xp102\\OneDrive\\바탕 화면\\time_series_COR_trend')
options(scipen = 999, digits=21)
# data<- read.csv('test_bigquery.csv') # nid가 정상적으로 읽히지 않는다.
data<- fread('test_bigquery_2.csv')

# 전처리 결과 불러오기 
source('df_prepro.R')
data<- df_prepro(data,seq_length = 36)


# COR
nid_list<- as.character(unique(data$nid))
COR<- matrix(0, nrow = length(nid_list), ncol = 34)
rownames(COR)<- nid_list


# trend 계산
library(forecast)

for(i in 1:length(nid_list) ){
  idx<- which(data$nid == nid_list[i])
  
  qq <- forecast::ma(data[idx,'pt'], order = 3)
  qq <- qq[!is.na(qq)]
  
  ii<-which(rownames(COR) %in% nid_list[i]) 
  
  COR[ii,]<-qq
}



# COR + trend 구하기
library(parallelDist) # https://www.rdocumentation.org/packages/parallelDist/versions/0.1.1/topics/parDist

source("corFuncPtr.R")
d<- parDist(COR, method="custom", func = corFuncPtr)
d[is.na(d)]<-0

# 군집

# Hierarchical clustering using Complete Linkage
hc1 <- hclust(d, method = "ward.D" )

# Plot the obtained dendrogram
plot(hc1, cex = 0.01, hang = -1, main= 'SZ 2019-08-31 누적매출 상위 15% 유저', label = FALSE)
rect.hclust(hc1, k = 3, border = 2:5)


# Cut tree into 3 groups
sub_grp <- cutree(hc1, k = 6)

# Number of members in each cluster
table(sub_grp)

# 시각화
library(ggplot2)
library(RColorBrewer)
library(reshape)
casted_data<- cast(data, nid ~ rn)
pt_matrix<- as.matrix(casted_data[, -1])

# cluster_1
group_num = 1
idx<- which(sub_grp== group_num)
idx<- which( as.character(data$nid) %in% names(sub_grp[idx]) )


temp<- data[idx,]
temp$nid<- as.factor(temp$nid)



p1<-ggplot(temp,  aes(rn, nid)) +
  geom_tile(aes(fill = pt)) +
  scale_fill_gradientn(colors = rev(brewer.pal(11, "RdBu"))) +
  scale_y_discrete(limits = rev(unique(temp$nid))) +
  ggtitle(paste('User Group',group_num, "by 10 minute")) +
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
  geom_line(size=2)+ ylim(0,350)+
  ggtitle("User Group Mean Play Time")+
    theme(axis.ticks.x=element_blank(),
          
          axis.title.x=element_blank())


library(gridExtra)
grid.arrange(p2, p1, nrow=2, top= paste('Group', group_num))



# 유저 특성 찾기
library(bigrquery)

# Provide authentication through the JSON service account key
path="D:/데이터분석/BigQuery/lgsz-0718-f37976ac9c5c.json"
#set_service_token(path)
bq_auth(path)

# Store the project id
projectid="lgsz-0718"

# Set your query
group_num = 2
idx<- which(sub_grp== group_num)
idx<- which(data$nid %in% names(sub_grp[idx]))
temp <-data[idx,]

sql <- paste0(" 
              with sample as (
              SELECT nid,  max_pl
              FROM sz_dw.f_user_map
              where  date_diff_reg = 0
              and nid in ('",  paste( unique(temp$nid), collapse = "','" ), "')"
              , ")
              
              select 1.0 * sum(max_pl) / count(distinct nid)
              from sample
              
              "
              )


# Run the query and store the data in a dataframe
tb <- bq_project_query(query=sql,x=projectid) 
df <- bq_table_download(tb)
print(df)