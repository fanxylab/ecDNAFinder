# ecDNAFinder
The [**ecDNAFinder**](https://github.com/fanxylab/ecDNAFinder.git) is used for finding ecDNA breakpoint and circle from long read bam, such as pacbio and nanopore sequencing.<br/>
The ecDNA detection include the following steps:
- fetching mapping read intervals and referred genomic locations
- detecting reads rearrangements to get the ecDNA candidate reads;
- constructing the breakpoint graph for each read; 
- concatenation of consensus fragments;
- annotating mapped genome coordinates;
- filtering and visualization of the circles.<br/>

___
## Installation
### Dependencies
<pre><code>'joblib >= 0.13.2',
'matplotlib >= 3.0.3',
'numpy >= 1.16.4',
'pandas >= 0.24.2',
'Cython >= 0.29.21',
'numba >= 0.50.1',
'pysam >=0.16.0.1',
</code></pre>

### User installation
- download: https://github.com/fanxylab/ecDNAFinder.git
- cd ecDNAFinder
- python setup.py install --user
___
## useage
**ecDNAFinder..py -h**<br/>
**usage:** MLkit.py [-h] [-V] {Common,Fetch,Search,Merge,Update,Filter,Circos,Auto} ...<br/>

### **1. positional arguments:**
<p> {Common,Fetch,Search,Merge,Update,Filter,Circos,Auto}</p>
<pre><code>    Common              The common parameters used for other models.
    Fetch               fatch reads information from bam file.
    Search              search breakpoint region from bed file.
    Merge               merge breakpoint region from bed file.
    Update              merge all breakpoint region in all samples.
    Filter              filter links from bed file.
    Circos              Circos visual for the target links file.
    Auto                the auto-processing for all
</code></pre>      

### **2. Example:**
sample.info.txt format:
```shell
sampleid
sample1
sample2
```
workpath:
```shell
input_workpath
├── bamfile
│   ├── sample1.bam
│   └── sample2.bam
```
<pre><code>python ecDNAFinder.py Auto 
	-f sample.info.txt  #sample information
	-i workpath         #the input directory, including the bam file.
	-bd bamdir          #the  bamfile directory built in input directory
	-o outputpath       #the output directory
	-n 6                #the parallel number 
</code></pre>       

