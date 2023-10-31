docker_build:
	docker build . -t rl_image

# docker_start:
# 	docker run --name rl -e DISPLAY={DISPLAY} --net=host -v ~/Desktop/Docker_shared/Representation\ Learning\ for\ Model-Predictive\ Exploration\ in\ Reinforcement\ Learning:/Docker_shared/Representation\ Learning\ for\ Model-Predictive\ Exploration\ in\ Reinforcement\ Learning \
# 	 -it --privileged rl_image

docker_start:
	docker run --name rl -it rl_image

docker_rm:
	docker container rm rl && \
	docker image rm rl_image

train:
	torchrun --nnodes 1 --nproc_per_node 6 --standalone main.py --train --config_path=./configs/demo_config.conf --log_name=demo_00 --save_model_path=checkpoints/demo_00.ckpt 


profile_training:
	torchrun --nnodes 1 --nproc_per_node 6 --standalone main.py --train --config_path=./configs/demo_config.conf --log_name=demo_00 --save_model_path=checkpoints/demo_00.ckpt --pytorch_profiling

test:
	torchrun --nnodes 1 --nproc_per_node 2 --standalone main.py --eval --config_path=./configs/demo_config.conf --log_name=demo_test_00 --load_model_path=checkpoints/demo_00.ckpt

kill:
	killall python3

start_tensorboard:
	tensorboard --logdir logs/tb_logs

start_tensorboard_profiles:
	tensorboard --logdir=./logs/torch_profiler_logs

run_tests:
	python3 test.py --train