#include "layer.h"
#include <stdlib.h>
#include <math.h>

// --- Helpers Privés ---

// Génère une valeur aléatoire entre -1 et 1
static Value randomValue() {
	return ((Value)rand() / (Value)RAND_MAX) * 2.0 - 1.0;
}

// Initialise les poids (W) avec une heuristique (Xavier/Glorot simplifié)
// Cela évite que les gradients explosent ou disparaissent lors des premières itérations.
static void initWeights(MatrixPtr W) {
	if (W->cols == 0) return;

	// Scale factor = 1 / sqrt(InputSize)
	Value scale = sqrt(1.0 / (Value)W->cols);

	for (SizeT i = 0; i < W->size; i++) {
		W->data[i] = randomValue() * scale;
	}
}

// --- Implémentation ---

void initLayer(LayerPtr layer, SizeT inputSize, SizeT outputSize, ActivationType activation) {
	layer->inputSize = inputSize;
	layer->outputSize = outputSize;
	layer->activation = activation;

	// 1. Allocation des Poids (Matrice Out x In)
	// createMatrix alloue et initialise à 0 (via calloc), mais on va remplir avec du random.
	layer->weights = createMatrix(outputSize, inputSize);
	initWeights(&layer->weights);

	// 2. Allocation des Biais (Vecteur Out x 1)
	// Initialisés à 0 par défaut, ce qui est une bonne pratique pour les biais.
	layer->biases = createVector(outputSize);

	// 3. Allocation des Caches (pour stocker les valeurs lors du Forward)
	layer->inputCache = createVector(inputSize);       // X (Entrée)
	layer->outputCache = createVector(outputSize);     // Z (Avant activation)
	layer->activationCache = createVector(outputSize); // A (Après activation)
}

void freeLayer(LayerPtr layer) {
	// Le 'false' indique qu'on ne demande pas de "Wipe" (mise à 0 sécurisée) des données avant le free.
	// Note : deallocMatrix ne libère que le buffer 'data', pas le pointeur 'weights' lui-même.
	deallocMatrix(&layer->weights, false);

	deallocVector(&layer->biases);
	deallocVector(&layer->inputCache);
	deallocVector(&layer->outputCache);
	deallocVector(&layer->activationCache);
}