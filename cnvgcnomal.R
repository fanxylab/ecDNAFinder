gcloessm <- function(Xy, x='gc', y='coverage', z='',
                 samplesize = 5000000, 
                 mappability = 0.9,
                 rep= 1000, 
                 xoutlier = c(0.001, 1-0.001),
                 youtlier = c(0, 1- 0.005),
                 aoutlier = c(0, 1- 0.01)){
    Xy <- as.data.frame(Xy)
    Xy$valid <- TRUE
    Xy$valid[Xy[[y]] <= 0 | Xy[[x]] < 0] <- FALSE

    xrange <- quantile(Xy[Xy$valid, x], prob = xoutlier, na.rm = TRUE)
    yrange <- quantile(Xy[Xy$valid, y], prob = youtlier, na.rm = TRUE)

    Xy$ideal <- TRUE
    Xy$ideal[!Xy$valid |
              Xy[[y]] <=yrange[1] | Xy[[y]] > yrange[2] |
              Xy[[x]] < xrange[1] | Xy[[x]] > xrange[2] ] <- FALSE

    if(z %in% colnames(Xy)){ Xy$ideal[ Xy[[z]] < mappability ] <- FALSE }

    i <- seq(0, 1, by = 0.001)
    sindex <- which(Xy$ideal)
    select <- sample(sindex, min(length(sindex), samplesize))
    lmodel <- loess(Xy[select, y] ~ Xy[select, x], span = 0.03)
    fmodel <- loess(predict(lmodel, i) ~ i, span = 0.3)
    Xy$copy <- Xy[[y]] / predict(fmodel, Xy[[x]])

    ###add mappability
    if(z %in% colnames(Xy)){
      arange <- quantile(Xy$copy[Xy$valid], prob = aoutlier, na.rm = TRUE)
      sindex <- which(Xy$copy < arange[2])
      select <- sample(sindex, min(length(sindex), samplesize))
      mmodel <- approxfun(lowess(Xy[select, z], Xy[select, 'copy']))
      Xy$copy <- Xy$copy / mmodel(Xy[[z]])
    }

    Xy$copy[Xy$copy <= 0 | abs(Xy$copy) == Inf ] = NA
    Xy$logcopy <- log(Xy$copy, 2)
    return(Xy)
}

gclowessu <- function(Xy, x='gc', y='coverage'){
    xraw <- Xy[[x]]
    yraw <- Xy[[y]]
    NONAIdx  <- which(!is.na(xraw) & !is.na(yraw) & yraw > 0)

    xflt <- Xy[NONAIdx, x] 
    yflt <- Xy[NONAIdx, y] 
    ypre <- rep(NA, length(yraw))

    Md <- lowess(xflt, log2(yflt), f=0.3)
    Ma <- approx(Md$x, Md$y, xraw)
    Ly <- log2(Xy[[y]]) - Ma$y
    Ly[abs(Ly)==Inf] = NA
    Xy$copy <- 2^Ly
    Xy$logcopy <- Ly
    return(Xy)
}

segmentcbs <- function(df, copy='copy', chrom='chrom', pos='start', sid='c05296'){
  library(DNAcopy)
  CNA.object <- CNA(df[copy],df[[chrom]],df[[pos]],
                    data.type="logratio",sampleid=sid)

  smoothed   <- smooth.CNA(CNA.object)
  segmented  <- segment(smoothed,
                      #undo.splits="sdundo", undo.SD=3, 
                      #alpha=, weights=
                      verbose=1)
  #segout <- cbind(segmented$output, segmented$segRows)
  segout <- segmented$output
  colnames(segout) <- c('SID', 'chrom', 'start', 'end', 'binsnum', 'meanlog2CN')
  segout <- segout[order(as.numeric(segout$chrom), segout$start),]
  return (segout)
}

segmenthmm <- function(logcopy){
  library(HMMcopy)
  default_param <- HMMsegment(logcopy, getparam = TRUE)
  longseg_param <- default_param
  longseg_param$e <- 0.7
  longseg_param$strength <- 100
  longseg_segments <- HMMsegment(logcopy, longseg_param)
  print(longseg_segments)
}

plotCorr <- function(df, X='gc', yraw='reads', ycor='copy', points = 10000, ...) {
  par(mfrow = c(1, 2))
  plot(df[[X]], df[[yraw]],
       col = densCols(df[[X]], df[[yraw]]),
       pch = 20, cex=0.1, 
       ylab = "Uncorrected Readcount", xlab = "GC content",
       main = "GC Bias in Uncorrected Readcounts", ...)

  plot(df[[X]], df[[ycor]],
       col = densCols(df[[X]], df[[ycor]]),
       pch = 8, cex=0.1, 
       ylab  = "corrected Readcount", xlab = "GC content",
       main = "GC Bias in Corrected Readcounts", ...)
}

findbin <- function(rbin, sbin){
  rbin$sstart <- rbin$start
  rbin$send   <- rbin$end
  rbin$meanlog2CN <- rbin$logcopy

  for (i in seq(dim(sbin)[1])){
    kidx = rbin$chrom == sbin$chrom[i] &
      rbin$start >= sbin$start[i] & 
      rbin$end   <= sbin$end[i]
    rbin$sstart[kidx] <- sbin$start[i]
    rbin$send[kidx]   <- sbin$end[i]
    rbin$meanlog2CN[kidx & !is.na(rbin$logcopy)] <- sbin$meanlog2CN[i]
  }
  
  rbin$CNtype <- NA
  rbin$CNtype[rbin$meanlog2CN<=0 & !is.na(rbin$logcopy)] <- 'HOMD'
  rbin$CNtype[rbin$meanlog2CN>0  & rbin$meanlog2CN<=1 &
                                   !is.na(rbin$logcopy)] <- 'HETD'
  rbin$CNtype[rbin$meanlog2CN>1  & rbin$meanlog2CN<3 &
                !is.na(rbin$logcopy)] <- 'NEUT'
  rbin$CNtype[rbin$meanlog2CN>=3  & rbin$meanlog2CN<4 &
                !is.na(rbin$logcopy)] <- 'GAIN'
  rbin$CNtype[rbin$meanlog2CN>=4  & rbin$meanlog2CN<5 &
                !is.na(rbin$logcopy)] <- 'AMPL'
  rbin$CNtype[rbin$meanlog2CN>=5 & !is.na(rbin$logcopy)] <- 'HLAMP'
  #HOMD Homozygous deletion, ≤ 0 copies
  #HETD Heterozygous deletion, 1 copy
  #NEUT Neutral change, 2 copies
  #GAIN Gain of chromosome, 3 copies
  #AMPL Amplification event, 4 copies
  #HLAMP High level amplification, ≥ 5 copies
  return (rbin)
}

plotCNV <-function(corseg, out, heig=4){
  library(ggplot2)
  library(ggnewscale)
  levec <- as.character(sort(as.numeric(unique(corseg$chrom))))
  corseg$chrom <- factor(corseg$chrom, levels=levec)
  if (length(unique(corseg$SID)) >1){
    leves <- as.character(sort(unique(corseg$SID)))
    corseg$SID <- factor(corseg$SID, levels=leves)
  }
  g <- ggplot(corseg, aes(x=start, y=logcopy)) + 
      geom_point(size=0.15, aes(color = logcopy) ) +
      scale_colour_viridis_c("meamlogcopy", begin=1, end=0.4) + 
      new_scale_colour() + 
      geom_step( aes(x = start, y = meanlog2CN, color=meanlog2CN), na.rm = TRUE) +
      scale_colour_gradientn(colours=c("blue", "red"))
  #scale_colour_gradientn(colours=c("green", "red"))
  #geom_segment(aes(sstart, meanlog2CN, xend=send, yend=meanlog2CN, color=meanlog2CN), size = 0.65) +
  g <- g + labs(x="", y="CNV_log2count profile\n")
  g <- g + facet_grid(SID~chrom, space="free_x", scales="free_x", margins=FALSE) 
  g <- g + theme(text= element_text(color = "black",size=11),
                 legend.text=element_text(color = "black",size = 11),
                 panel.background=element_rect(color="black"),
                 axis.text.x = element_text(angle = 90), 
                 #panel.grid.minor = element_blank(), 
                 #panel.grid.major = element_blank(),
                 axis.title.x=element_blank())
  heig <-  1.1*(length(unique(corseg$SID)) -1) + 4
  ggsave(out, g, width=28, height = heig )
}

plotCNVM <-function(corseg, out, heig=4){
  library(ggplot2)
  library(ggnewscale)
  levec <- as.character(sort(as.numeric(unique(corseg$chrom))))
  corseg$chrom <- factor(corseg$chrom, levels=levec)
  if (length(unique(corseg$SID)) >1){
    leves <- as.character(sort(unique(corseg$SID)))
    corseg$SID <- factor(corseg$SID, levels=leves)
  }
  g <- ggplot(corseg, aes(x=start, y=logcopy)) + 
      geom_point(size=0.15, aes(color = logcopy) ) +
      scale_colour_viridis_c("meamlogcopy", begin=1, end=0.4) + 
      new_scale_colour() + 
      geom_step( aes(x = start, y = meanlog2CN, color=meanlog2CN, group=SID), na.rm = TRUE) +
      scale_colour_gradientn(colours=c("blue", "red"))

  #scale_colour_gradientn(colours=c("green", "red"))
  #geom_segment(aes(sstart, meanlog2CN, xend=send, yend=meanlog2CN, color=meanlog2CN), size = 0.65) +
  g <- g + labs(x="", y="CNV_log2count profile\n")
  g <- g + facet_grid(. ~chrom, space="free_x", scales="free_x", margins=FALSE) 
  g <- g + theme(text= element_text(color = "black",size=11),
                 legend.text=element_text(color = "black",size = 11),
                 panel.background=element_rect(color="black"),
                 axis.text.x = element_text(angle = 90), 
                 #panel.grid.minor = element_blank(), 
                 #panel.grid.major = element_blank(),
                 axis.title.x=element_blank())
  #heig <-  1.1*(length(unique(corseg$SID)) -1) + 4
  heig <-  5
  ggsave(out, g, width=28, height = heig )
}

getGC_seg <-function(data, hd, yname='counts'){
    ####################GC
    cornom1 <- gcloessm(data, x='gc',y=yname)
    cornom2 <- gclowessu(data, x='gc',y=yname) # when big data, bias is still exists
    write.table(cornom1, paste0(hd, '.cor.nom.txt'), sep='\t', row.names = FALSE, na='', quote=FALSE)

    pdf(file = paste0(hd, '.cor.nom.pdf'), width = 8, height = 4)
    plotCorr(cornom1, X='gc', yraw=yname, ycor='logcopy')
    dev.off()

    #########################segmentation
    segdata <- cornom1[,c('chrom', 'start', 'end', 'logcopy')]
    colnames(segdata) <- c('chrom', 'start', 'end', 'copy')
    segdata$chr <- as.factor(segdata$chr)
    segcbs <- segmentcbs(segdata, sid=SID) #segmenthmm(segdata)
    corseg <- findbin(cornom1, segcbs)
    write.table(corseg, paste0(hd, '.cor.seg.txt'), sep='\t', row.names = FALSE, na='', quote=FALSE)
    ######################plot
    # To use for fills, add
    #plotCNV(corseg, paste0(hd, '.cor.seg.pdf'))
    plotCNV(corseg, paste0(hd, '.cor.seg.png'))
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
    return(data)
}

mergemean <-function(adata, method='mean'){
    library(dplyr)
    bdata = adata %>% 
            group_by(chrom,	start,	end,	length,	gc,	rmsk,	bins) %>%
            summarize(  SID = paste(unique(SID), collapse=';'),
                        copy= mean(copy, na.rm = TRUE),
                        logcopym = mean(logcopy, na.rm = TRUE),
                        logcopy  = log( mean(copy, na.rm = TRUE), 2)
                        #logcopym<=logcopy #am-gm inequality
            )
    bdata = as.data.frame(bdata)
    return(bdata)
}

mergeseg <-function(cornom1, copycol='logcopy'){
    segdata <- cornom1[,c('chrom', 'start', 'end', copycol)]
    colnames(segdata) <- c('chrom', 'start', 'end', 'copy')
    segdata$chr <- as.factor(segdata$chr)
    segcbs <- segmentcbs(segdata, sid='Mergeall') #segmenthmm(segdata)
    corseg <- findbin(cornom1, segcbs)
    return(corseg)
}

args <-commandArgs(T)
wd   <- args[1]
hd   <- args[2]
ty   <- args[3]

if ( !is.na(ty) & !ty %in% c('each', 'all')){
    message('The third argument is error.')
    quit(status=1)
}

if ( is.na(ty) | ty =='each'){
    #wd workdir
    #hd header with out .gz
    setwd(wd)
    IN <- paste0(hd, '.gz')
    IN <- gzfile(IN,'rt')
    data <- read.csv(IN, sep='\t')
    close(IN)

    data <- changechr(data, keep=c('X'))
    SID  <- data$SID[1]
    getGC_seg(data, hd)
    if ('counts_md' %in% colnames(data)){
        data$counts_md[data$counts_md==0] <- NA
        getGC_seg(data, paste0(hd, '.md'), yname='counts_md')
    }
}else if(ty =='all'){
    #setwd(wd)
    #system( paste('ls', k), intern = TRUE)
    #wd out file
    #hd all "cor.seg.txt" file split with ";"
    hd <- unlist(strsplit(hd, split =';'))
    print(hd)
    adata <- lapply(hd, function(x) if(file.exists(x)){read.csv(x, sep='\t', header=TRUE)})
    adata <- Reduce(rbind, adata)
    write.table(adata, paste0(wd, '.xls'), sep='\t', quote=FALSE, row.names=FALSE)
    plotCNV( adata, paste0(wd, '.each.pdf'), heig=12)
    plotCNVM(adata, paste0(wd, '.all.pdf'), heig=12)

    bdata <- mergemean(adata)
    cdata <- mergeseg(bdata, copycol='logcopy')
    write.table(cdata, paste0(wd, '.logmean.xls'), sep='\t', quote=FALSE, row.names=FALSE)
    plotCNV( cdata, paste0(wd, '.logmean.pdf'), heig=12)
}

