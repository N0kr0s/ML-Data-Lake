.PHONY: venv install model all


# Устанавливает зависимости через uv pip (опционально, добавьте сюда ваши библиотеки)
install:
	uv sync

# Скачивает и устанавливает языковую модель spaCy
model: install
	uv run spacy download en_core_web_trf

# Создаёт виртуальное окружение через uv (если не создано)
init:
	uv init

# Полная инициализация
all: model
