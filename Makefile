# Makefile

# Define o alvo 'format' para rodar o black com limite de 99 caracteres por linha
format:
	black -l 99 .

# Define o alvo 'check' para verificar a formatação sem aplicar mudanças
check:
	black -l 99 --check .

# Alvo padrão que roda a formatação
.PHONY: format check
