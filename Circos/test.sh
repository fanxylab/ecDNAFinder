circos  -conf ./links.circos.conf -outputdir . -outputfile link.site.svg

rsvg-convert link.site.svg -f pdf -o link.site.pdf

