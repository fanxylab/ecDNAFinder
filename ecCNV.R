plotCNV2 <-function(corseg, out){
  library(ggplot2)
  library(ggnewscale)
  #library(colormap)
  levec <- as.character(sort(as.numeric(unique(corseg$chrom))))
  levec[levec=='23'] = 'X'
  levec[levec=='24'] = 'Y'
  levec[levec=='25'] = 'MT'
  panel_color = as.character(as.numeric(corseg$chrom) %% 2)
  #panel_color$chrom[panel_color$chrom=='23'] = 'X'
  #panel_color$start = panel_color$copy = 1
  #panel_color$chrom <- factor(panel_color$chrom, levels=levec)
  copy_color = corseg$chrom
  copy_color = as.numeric(copy_color) %% 2
  #copy_color[is.na(copy_color)] = 0
  #copy_color[copy_color > ylimit] = ylimit
  corseg$chrom[corseg$chrom =='23' ] = 'X'
  corseg$chrom[corseg$chrom =='24' ] = 'Y'
  corseg$chrom[corseg$chrom =='25'] = 'MT'
  corseg$chrom <- factor(corseg$chrom, levels=levec)
  ylimit = 4 * floor(mean(corseg$copy, na.rm=TRUE))
  ylimit = 4 * floor(median(corseg$copy, na.rm=TRUE))
  ylimit = 8
  #copy_color = corseg$copy
  #copy_color[is.na(copy_color)] = 0
  #copy_color[copy_color > ylimit] = ylimit
  mean_color = corseg$meanCN
  mean_color[mean_color > ylimit] = ylimit
  mean_color = 0
  g = ggplot(corseg, aes(x=start, y=copy))
  g = g + geom_point(size=1, aes(color=factor(copy_color)), show.legend = F)
  g = g + scale_color_manual(values = c("#FD6D4E", "#71D3F2"))
  #g = g + scale_color_brewer(palette = 'Accent')
  #g = g + scale_colour_gradientn("mean CN", colours=c("#66CCCC","#E66800"))
  g = g + geom_point(aes(x=start, y=meanCN), color="black", size=1, na.rm = TRUE)
  g = g + facet_grid(1~chrom, space="free_x", scales="free_x", margins=FALSE) 
  g = g + theme(#panel.background=element_rect(fill="lightblue"),
                text=element_text(size=rel(5)),
                legend.text=element_text(size=rel(3)),
                #strip.text.x=element_text(size=rel(3)),
                strip.text = element_blank(),
                strip.background = element_blank(),
                #strip.background = element_rect(fill="white"),
                panel.grid.minor = element_blank(), 
                panel.grid.major = element_blank(),
                panel.background = element_rect(fill="white",color="grey70", size=2),
                #panel.grid.major.x = element_blank(),
                #panel.grid.major.y = element_line(size=0.5),
                panel.spacing.x = unit(-1, 'lines'),
                #plot.background = element_rect(colour = "grey50", size = 2),
                axis.text.x = element_blank(), 
                axis.ticks.x = element_blank(), 
                axis.title.x = element_blank())
  g = g + labs(x="", y="")
  g = g + scale_x_continuous(expand=c(0,0))
  g = g + scale_y_continuous(limits=c(0,ylimit), expand=c(0,0))
  write.table(corseg, gsub('pdf', 'cnv', out), sep='\t', row.names = FALSE, na='', quote=FALSE)
  ggsave(out, g, width=28, height=4)
}

plotCNV <-function(corseg, out){
  library(ggplot2)
  library(ggnewscale)
  #library(colormap)
  levec <- as.character(sort(as.numeric(unique(corseg$chrom))))
  levec[levec=='23'] = 'X'
  levec[levec=='24'] = 'Y'
  levec[levec=='25'] = 'MT'
  panel_color = unique(corseg['chrom'])
  panel_color$color_id = as.character(as.numeric(panel_color$chrom) %% 2)
  panel_color$chrom[panel_color$chrom=='23'] = 'X'
  panel_color$start = panel_color$copy = 1
  panel_color$chrom <- factor(panel_color$chrom, levels=levec)
  corseg$chrom[corseg$chrom =='23' ] = 'X'
  corseg$chrom[corseg$chrom =='24' ] = 'Y'
  corseg$chrom[corseg$chrom =='25'] = 'MT'
  corseg$chrom <- factor(corseg$chrom, levels=levec)
  ylimit = 4 * floor(mean(corseg$copy, na.rm=TRUE))
  ylimit = 4 * floor(median(corseg$copy, na.rm=TRUE))
  copy_color = corseg$copy
  copy_color[is.na(copy_color)] = 0
  copy_color[copy_color > ylimit] = ylimit
  mean_color = corseg$meanCN
  mean_color[mean_color > ylimit] = ylimit
  g = ggplot(corseg, aes(x=start, y=copy))
  g = g + geom_rect(data=panel_color, aes(fill=color_id),xmin = -Inf,xmax = Inf,ymin = -Inf,ymax = Inf,alpha = 0.3, show.legend=FALSE)
  #g = g + scale_fill_manual(values=c("1"="red", "0"="blue")) # 改变框颜色
  g = g + geom_point(size=0.15, aes(color=copy_color))
  #g = g + scale_colour_gradientn("mean CN", colours=c("blue", "red", "darkred"))
  g = g + scale_colour_gradientn("mean CN", colours=c("darkblue","blue", "red", "darkred", 'black'))
  #g = g + new_scale_colour()
  g = g + geom_point(aes(x=start, y=meanCN, color=mean_color), size=0.5, na.rm = TRUE)
  #g = g + geom_step(aes(x=start, y=meanCN, color=mean_color), size=1, na.rm = TRUE)
  #g = g + scale_colour_gradientn("mean CN", colours=c("blue", "red"))
  g = g + facet_grid(1~chrom, space="free_x", scales="free_x", margins=FALSE) 
  g = g + theme(#panel.background=element_rect(fill="lightblue"),
                text=element_text(size=rel(5)),
                legend.text=element_text(size=rel(3)),
                strip.text.x=element_text(size=rel(3)),
                strip.background=element_rect(fill="white"),
                panel.grid.minor = element_blank(), 
                panel.grid.major = element_blank(),
                #panel.grid.major.x = element_blank(),
                #panel.grid.major.y = element_line(size=0.5),
                panel.spacing.x = unit(0.1, 'lines'),
                axis.text.x = element_blank(), 
                axis.ticks.x = element_blank(), 
                axis.title.x=element_blank())
  g = g + labs(x="", y="CNV count profile\n")
  g = g + scale_x_continuous(expand=c(0,0))
  g = g + scale_y_continuous(limits=c(0,ylimit), expand=c(0,0))
  #write.table(corseg, gsub('png', 'txt', out), sep='\t', row.names = FALSE, na='', quote=FALSE)
  ggsave(out, g, width=28, height=4)
}

ez_cnv <- function(data) {
  library(DNAcopy)
  l = dim(data)[1]
  m = mean(data$counts_md + 1, na.rm = TRUE)
  normal = (data$counts_md + 1) / m
  F = normal
  low = lowess(data$gc, log(data$counts_md / m))
  z = approx(low$x, low$y, data$gc)
  normal = exp(log(data$counts_md / m) - z$y)
  lr = log2(normal/F)
# Determine breakpoints and extract chrom/locations
  CNA.object = CNA(genomdat = lr, chrom = data$chrom, maploc = as.numeric(data$end), data.type = 'logratio')
  CNA.smoothed = smooth.CNA(CNA.object)
  segs = segment(CNA.smoothed, verbose=0, min.width=5)
  frag = segs$output[,2:3]

# Map breakpoints to kth sample
  len = dim(frag)[1]
  bps = array(0, len)
  for (j in 1:len)
    bps[j] = which((data$chrom==frag[j,1]) & (as.numeric(data$end)==frag[j,2]))
  bps = sort(bps)
  bps[(len=len+1)] = l
# Track global breakpoint locations
  breaks = matrix(0, l)
  breaks[bps] = 1
# Modify bins to contain median read count/bin within each segment
  fixed = matrix(0, l)
  fixed[1:bps[2]] = median(normal[1:bps[2]], na.rm=TRUE)
  for(i in 2:(len-1))
    fixed[bps[i]:(bps[i+1]-1)] = median(normal[bps[i]:(bps[i+1]-1)], na.rm=TRUE)
  fixed = fixed/mean(fixed, na.rm=TRUE)

# Determine Copy Number
  CNgrid = seq(2, 6, by=0.05)
  outerRaw = fixed %o% CNgrid
  outerRound = round(outerRaw)
  outerDiff = (outerRaw - outerRound) ^ 2
  outerColsums = sum(outerDiff, na.rm = FALSE)
  CNmult = CNgrid[order(outerColsums)]
  CNerror = round(sort(outerColsums), digits=2)
  CN = CNmult[1]
  final = round(fixed * CN)
  final_nr = fixed * CN
  cloud = normal * CN
  data$copy = cloud
  data$meanCN = final
  return(data)
}

changechr <-function(data, keep=c('X','Y', 'MT')){
    chroms <- c(as.character(seq(1,22)), keep)
    data  <- data[data$chrom %in% chroms,]
    data$chrom <- as.character(data$chrom)
    if ('X'  %in% keep){data$chrom[data$chrom =='X' ] = '23'}
    if ('Y'  %in% keep){data$chrom[data$chrom =='Y' ] = '24'}
    if ('MT' %in% keep){data$chrom[data$chrom =='MT'] = '25'}
    data <- data[order(as.numeric(data$chrom), data$start),]
    data$counts[data$counts==0] <- NA
    data$counts_md[data$counts_md==0] <- NA
    return(data)
}

args <-commandArgs(T)
out  <- args[1]
hd   <- args[2]
setwd(out)
data <- read.csv(hd, sep='\t')
data <- changechr(data, keep=c('X')) # only keep x chrosome
t = strsplit(out, '/')[[1]]
out_png = sprintf('%s/%s.pdf', out, t[length(t)])
data = ez_cnv(data)
#plotCNV(data, out_png)
plotCNV2(data, out_png)
