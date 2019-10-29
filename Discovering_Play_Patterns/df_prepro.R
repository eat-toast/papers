library(dplyr)


df_prepro<- function(df, seq_length = 36){
  colum_name<- colnames(df)
  nid_list<- unique(df$nid)
  
  #nid_and_seq<- merge(nid_list, seq_len(seq_length), all.x = True)
  # nid_and_seq<- nid_and_seq%>%arrange(x, y)
  nid_and_seq<- data.table::CJ(nid = nid_list, rn = seq_len(seq_length))
  nid_and_seq<- nid_and_seq%>%arrange(nid, rn)
  
  df<- nid_and_seq%>%left_join(df, by = c('rn' = 'rn', 'nid' = 'nid') )
  
  # 결측치 0으로 대체
  df$pt[is.na(df$pt)]<-0
  
  # 데이터 최종 정리
  colnames(df)<- colum_name
  
  return(df)
}
