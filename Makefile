SHELL := /bin/bash

install:
	pip install -r requirements.txt	

test:
	pytest tests
