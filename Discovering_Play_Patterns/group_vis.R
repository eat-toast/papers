
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