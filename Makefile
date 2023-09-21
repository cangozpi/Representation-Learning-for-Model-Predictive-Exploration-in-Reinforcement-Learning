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
	python3 main.py --train

test:
	python3 main.py --eval

kill:
	killall python3

start_tensorboard:
	tensorboard --logdir logs/tb_logs