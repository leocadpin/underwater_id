#!/bin/bash

echo "Calculando resultados de segmentacion dados por la version 1"

meanIoU=0
count=0

# shopt -s cdable_vars
# ln -s /home/leo/Escritorio/Uware/1_imagenes/underwater_id ~/pathABC


for i in {1..8}
do
    # cd ~/pathABC
    # echo "$PATH"
	filename=$(basename "$i")
	# # n=${filename%.*}

	python3 detection.py -i $i -n $i
	IoU="$(python3 acierto.py -n $i)"

	echo $filename = $IoU
	meanIoU=`echo $meanIoU+$IoU|bc`
	let "count=count+1"
done

echo "-------"
echo -n "Media IoU = "
echo "scale=5; $meanIoU/$count"|bc -l|sed 's/^\./0./'
