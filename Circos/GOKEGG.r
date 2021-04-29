library(DOSE)
library(org.Hs.eg.db)
library(topGO)
library(clusterProfiler)
library(pathview)
keytypes(org.Hs.eg.db) 

INFile <- '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/COLON/Circos/20210405/data/cancer.gene.txt'
geneFile <- read.csv(INFile, header=F)
OUTDir <- '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/COLON/20210405/GoKegg'
dir.create(OUTDir, showWarnings = TRUE, recursive = TRUE)
setwd(OUTDir)

Genes <- bitr(geneFile[,1], 
            fromType="SYMBOL",
            toType=c("ENSEMBL", "ENTREZID"), 
            OrgDb="org.Hs.eg.db")
head(Genes,2)

###########backgroundGeneset
#data(geneList, package="DOSE") #富集分析的背景基因集
#gene <- names(geneList)[abs(geneList) > 2]
#gene.df <- bitr(gene, fromType = "ENTREZID", toType = c("ENSEMBL", "SYMBOL"), OrgDb = org.Hs.eg.db)
#head(gene.df,2)


#####################GO
ggo <- groupGO(gene = Genes$ENTREZID, 
              OrgDb = org.Hs.eg.db,
              ont = "CC",
              level = 3,
              readable = TRUE)

ego_ALL <- enrichGO(gene = Genes$ENTREZID, 
                    #universe = names(geneList),
                    OrgDb = 'org.Hs.eg.db', #organism="human"
                    keyType = 'ENTREZID',
                    ont = "ALL", #CC, BP, MF 
                    pAdjustMethod = "BH",
                    pvalueCutoff =0.05,
                    qvalueCutoff = 0.2,
                    readable = TRUE) #Gene ID to gene Symbol

head(ego_ALL,2)
dim(ego_ALL[ego_ALL$ONTOLOGY=='BP',])
dim(ego_ALL[ego_ALL$ONTOLOGY=='CC',])
dim(ego_ALL[ego_ALL$ONTOLOGY=='MF',])


dotplot(ego_ALL,title="EnrichmentGO_ALL_dot", showCategory=20)
barplot(ego_ALL, showCategory=20,title="EnrichmentGO_ALL", drop=T)
cnetplot(ego_ALL, title="EnrichmentGO_ALL_net", circular = TRUE, colorEdge =TRUE)

ggsave(paste("./GO.DO_DGN.plot.",c1,".pdf",sep=''), 
       grid.arrange(grobs = list(pl.d1, pl.d2, pl.c1,pl.c2), ncol = 2), 
       width = 18, height = 15)

#######################KEGG
kegg <- enrichKEGG(gene = Genes$ENTREZID,
                   organism = 'hsa',  ## hsa为人的简写，bta是牛的简写 
                   keyType = 'kegg', 
                   pvalueCutoff = 0.05,
                   pAdjustMethod = 'BH', 
                   minGSSize = 3,
                   maxGSSize = 500,
                   qvalueCutoff = 0.2,
                   use_internal_data = FALSE)
dotplot(kegg, title="KEGG_ALL_dot", showCategory=20) #气泡图
barplot(kegg, title="KEGG_ALL_bat", showCategory=20,drop=T) #柱状图
cnetplot(kegg, title="EnrichmentGO_ALL_net", circular = TRUE, colorEdge =TRUE)

cnetplot(kegg) #网络图
heatplot(kegg) #热力图
