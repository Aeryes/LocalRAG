.PHONY: up down test logs

up:
	# Automatically create the shared directory on the host to prevent Docker root-ownership issues
	mkdir -p shared_docs
	docker-compose up -d --build

down:
	docker-compose down

logs:
	docker-compose logs -f app

test:
	# Runs the DeepEval unit tests inside the container
	docker-compose exec app pytest tests/