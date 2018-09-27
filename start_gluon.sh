container='xuan_gluoncv'
image='anxu829/gluoncv:withlab'
sudo docker stop $container 
sudo docker rm $container
sudo nvidia-docker run --name $container  -e CUDA_VISIBLE_DEVICES=0  -it \
           -v /data/el-train/TRAIN/mxnet_yinlieClassify:/xuan \
           -v /data/el-train/TRAIN/xuhan_train/:/xuhan \
           -v /data/el-train/:/el-train \
	   	   -p $1:$1 $image \
		   	   /bin/bash \
			   	   -c " cd / && jupyter lab --port $1  --ip 0.0.0.0 --allow-root"
				   	    


