args <-commandArgs(T)
model = args[1]
info = args[2]
out = args[3]

data = read.csv(model, sep='\t')
data$counts_md[data$counts_md==0] <- NA
m = mean(data$counts_md + 1, na.rm = TRUE)
normal = (data$counts_md + 1) / m
low = lowess(data$gc, log(data$counts_md / m))

scale = read.csv(info, sep='\t')
z_scale = approx(low$x, low$y, scale$gc)
n_scale = exp(log(scale$scaled_depth / m) - z_scale$y)

scale$copy = n_scale
names(scale)[1] = '#chrom'
write.table(scale, out, sep='\t', row.names = FALSE, na='', quote=FALSE)
