CC = gcc
CFLAGS = -O3 -Wall -Wextra -Iinclude -Iexternal/myOwnCLib
LDFLAGS = -lm

# --- Sources ---

AI_SRCS = src/network.c \
          src/layer.c

CLIB_SRCS = external/myOwnCLib/maths/matrices/matrix.c \
            external/myOwnCLib/maths/matrices/mlMatrix.c \
            external/myOwnCLib/maths/vectors/vectors.c

ALL_SRCS = $(AI_SRCS) $(CLIB_SRCS)

# --- Cibles ---

.PHONY: all clean test

# 'make all' compile tout mais ne lance rien
all: test_xor test_forward test_save_load

# --- Règles de Compilation ---

test_xor:
	$(CC) $(CFLAGS) tests/test_xor.c $(ALL_SRCS) -o test_xor $(LDFLAGS)

test_forward:
	$(CC) $(CFLAGS) tests/test_forward.c $(ALL_SRCS) -o test_forward $(LDFLAGS)

test_save_load:
	$(CC) $(CFLAGS) tests/test_save_load.c $(ALL_SRCS) -o test_save_load $(LDFLAGS)

# --- Règle d'Exécution Globale ---

# 'make test' compile tout (si nécessaire) puis lance les exécutables
test: all
	@echo "\n==================================="
	@echo "    🚀 EXÉCUTION DES TESTS"
	@echo "==================================="
	@echo "\n--- [ 1/3 ] Test Forward ---"
	@./test_forward
	@echo "\n--- [ 2/3 ] Test Save/Load ---"
	@./test_save_load
	@echo "\n--- [ 3/3 ] Test XOR (Training) ---"
	@./test_xor
	@echo "\n✅ TOUS LES TESTS SONT TERMINÉS."

# --- Nettoyage ---

clean:
	rm -f test_xor test_forward test_save_load *.bin