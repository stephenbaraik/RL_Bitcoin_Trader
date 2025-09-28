build:
	# TAB before docker-compose!
	docker-compose -f compose/docker-compose.yml build

up:
	docker-compose -f compose/docker-compose.yml up -d

down:
	docker-compose -f compose/docker-compose.yml down

train:
	docker-compose -f compose/docker-compose.yml run --rm trainer
