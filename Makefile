# ==========================================
# Makefile pour myOwnAILib
# ==========================================

# Compilateur et Options
CC = gcc
# -O3 : Optimisation maximale (Crucial pour les perfs ML)
# -Iinclude : Pour inclure directement "network.h" qui est dans le dossier include/
# -Iexternal/myOwnCLib : Pour pouvoir inclure <maths/matrices/matrix.h>
CFLAGS = -O3 -Wall -Wextra -Iinclude -Iexternal/myOwnCLib

# Bibliothèques à lier (Maths standard)
LDFLAGS = -lm

# Dossiers
CLIB_DIR = external/myOwnCLib
SRC_DIR = src
TEST_DIR = tests

# ==========================================
# 1. SOURCES
# ==========================================

# -> Sources du Moteur (myOwnCLib)
# On sélectionne uniquement les fichiers mathématiques nécessaires
SRCS_CLIB = $(CLIB_DIR)/maths/matrices/matrix.c \
            $(CLIB_DIR)/maths/matrices/mlMatrix.c \
            $(CLIB_DIR)/maths/vectors/vectors.c 

# -> Sources de l'Application (myOwnAILib)
# Tous les fichiers .c dans src/ (network.c, layer.c, etc.)
SRCS_AI = $(wildcard $(SRC_DIR)/*.c)

# Tous les objets à compiler (.c -> .o)
OBJS = $(SRCS_CLIB:.c=.o) $(SRCS_AI:.c=.o)

# Nom de la librairie statique finale
TARGET_LIB = libmyownailib.a

# ==========================================
# 2. RÈGLES
# ==========================================

.PHONY: all clean test check_submodule

all: check_submodule $(TARGET_LIB)

# Création de l'archive statique (.a)
$(TARGET_LIB): $(OBJS)
	@echo "📚 Archiving library $@..."
	ar rcs $@ $^
	@echo "✅ Library created successfully!"

# Compilation générique des .c en .o
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Règle utilitaire pour vérifier que le submodule est là
check_submodule:
	@if [ ! -f "$(CLIB_DIR)/maths/matrices/matrix.c" ]; then \
		echo "❌ Erreur: myOwnCLib introuvable ou vide."; \
		echo "👉 Lancez: git submodule update --init --recursive"; \
		exit 1; \
	fi

# ==========================================
# 3. TESTS
# ==========================================

# Compile et lance chaque fichier de test individuellement
test: $(TARGET_LIB)
	@mkdir -p bin
	@echo "🧪 Running Test Suite..."
	@for file in $(TEST_DIR)/*.c; do \
		test_name=$$(basename $$file .c); \
		echo "--------------------------------------------------"; \
		echo "🔨 Compiling $$test_name..."; \
		$(CC) $(CFLAGS) $$file -L. -lmyownailib $(LDFLAGS) -o bin/$$test_name; \
		if [ $$? -eq 0 ]; then \
			echo "🚀 Running $$test_name..."; \
			./bin/$$test_name; \
		else \
			echo "❌ Compilation failed for $$test_name"; \
			exit 1; \
		fi; \
	done
clean:
	rm -f $(OBJS) $(TARGET_LIB)
	rm -rf bin
	@echo "🧹 Cleaned up."