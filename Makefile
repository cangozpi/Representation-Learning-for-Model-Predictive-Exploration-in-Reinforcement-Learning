docker_build:
	docker build . -t rl_image

docker_start:
	docker run --name rl -e DISPLAY={DISPLAY} --net=host -v ~/Desktop/Docker_shared/Representation\ Learning\ for\ Model-Predictive\ Exploration\ in\ Reinforcement\ Learning:/Docker_shared/Representation\ Learning\ for\ Model-Predictive\ Exploration\ in\ Reinforcement\ Learning \
	 -it --privileged rl_image


docker_start2:
	docker run --name rl -it rl_image

dev:
	docker container rm rl && \
	docker image rm rl_image && \

train:
	python3 train.py

kill:
	killall python3