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
	$(CC) $(CFLAGS) tests/test_xor.c $(ALL_SRCS) -o test_xor.out $(LDFLAGS)
	$(CC) $(CFLAGS) tests/test_xor.c $(ALL_SRCS) -S
	@./test_xor.out

test_forward:
	$(CC) $(CFLAGS) tests/test_forward.c $(ALL_SRCS) -o test_forward.out $(LDFLAGS)
	$(CC) $(CFLAGS) tests/test_forward.c $(ALL_SRCS) -S
	@./test_forward.out

test_save_load:
	$(CC) $(CFLAGS) tests/test_save_load.c $(ALL_SRCS) -o test_save_load.out $(LDFLAGS)
	$(CC) $(CFLAGS) tests/test_save_load.c $(ALL_SRCS) -S
	@./test_save_load.out

test_color_guess:
	$(CC) $(CFLAGS) tests/test_color_guess.c $(ALL_SRCS) -o test_color_guess.out $(LDFLAGS)
	$(CC) $(CFLAGS) tests/test_color_guess.c $(ALL_SRCS) -S
	@./test_color_guess.out

# --- Règle d'Exécution Globale ---

# 'make test' compile tout (si nécessaire) puis lance les exécutables
test: all
	@echo "\n==================================="
	@echo "    🚀 EXÉCUTION DES TESTS"
	@echo "==================================="
	@echo "\n--- [ 1/4 ] Test Forward ---"
	@./test_forward.out
	@echo "\n--- [ 2/4 ] Test Save/Load ---"
	@./test_save_load.out
	@echo "\n--- [ 3/4 ] Test XOR (Training) ---"
	@./test_xor.out
	@echo "\n--- [ 4/4 ] Test Color Guess (Training) ---"
	@./test_color_guess.out
	@echo "\n✅ TOUS LES TESTS SONT TERMINÉS."

# --- Nettoyage ---


clean:
	rm -f test_* *.bin *.s