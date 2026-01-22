.PHONY: up down test logs

up:
	docker-compose up -d --build

down:
	docker-compose down

logs:
	docker-compose logs -f app

test:
	# Runs the DeepEval unit tests inside the container
	docker-compose exec app pytest tests/