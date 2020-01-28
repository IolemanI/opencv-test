server: server.py
	python server.py

run: app.py
	python app.py

train: extract_embeddings.py train_model.py
	python extract_embeddings.py
	python train_model.py

dataset: create_dataset.py
	mkdir resources/dataset/${NAME}
	python create_dataset.py -n ${NAME}