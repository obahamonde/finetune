.PHONY: d, dev
dev:
	python3 -mprisma db push && python3 -m prisma generate && python3 -m prisma py fetch
	uvicorn main:app --host 0.0.0.0 --port 8888 --reload

.PHONY: b, build
build:
	docker compose up -d --build --remove-orphans

.PHONY: p, publish
publish:
	docker login
	docker compose build -t obahamondev/cuda-spark-runtime:latest .
	docker tag obahamondev/cuda-spark-runtime:latest obahamondev/cuda-spark-runtime:latest
	docker push obahamondev/cuda-spark-runtime:latest

