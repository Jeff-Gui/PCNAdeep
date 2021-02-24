dt = read.csv('/Users/jefft/Desktop/Chan lab/SRTP/ImageAnalysis/PCNAdeep/ccnbDeep/example/out/synchrogram/measure.csv')
library(tidyverse)
library(gridExtra)

dt$dist = as.factor(dt$dist)
for(t in unique(dt$trackId)){
  sub_table = subset(dt, dt$trackId==t)
  
  # plot all frames together
  ggplot(sub_table, aes(x=dist,y=mean, group=frame)) +
    theme_classic() +
    geom_line(aes(color=frame)) + 
    scale_color_gradient2(low='white',mid='red',high='#4B0082', midpoint=0)
    
  # plot each frame individually
  plot = lapply(unique(sub_table$frame), function(x){
    ggplot(sub_table %>% filter(frame==x)) + 
      geom_line(aes(x=dist,y=mean,group=frame)) + theme_classic() + labs(title=x) +
      theme(axis.title.x = element_blank(), axis.title.y = element_blank())
  })
  grid.arrange(grobs = plot, ncol=4)
}
