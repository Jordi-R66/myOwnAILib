#include "layer.h"
#include <stdlib.h>
#include <math.h>

// --- Helpers Privés ---

// Génère une valeur aléatoire entre -1 et 1
static Value randomValue() {
	return ((Value)rand() / (Value)RAND_MAX) * 2.0 - 1.0;
}

// Initialise les poids (W) avec une heuristique simple (type Xavier/Glorot simplifié)
// pour éviter la saturation des gradients au début.
static void initWeights(MatrixPtr W) {
	if (W->cols == 0) return;

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
	layer->weights = createMatrix(outputSize, inputSize);
	initWeights(&layer->weights);

	// 2. Allocation des Biais (Vecteur Out x 1)
	// createVector utilise calloc en interne via createMatrix, donc initialisé à 0.0
	layer->biases = createVector(outputSize);

	// 3. Allocation des Caches (Vecteurs Out x 1 ou In x 1)
	// Ils serviront à stocker les résultats intermédiaires pour la Backprop
	layer->inputCache = createVector(inputSize);       // X
	layer->outputCache = createVector(outputSize);     // Z
	layer->activationCache = createVector(outputSize); // A
}

void freeLayer(LayerPtr layer) {
	// Le second paramètre 'false' indique qu'on ne libère pas le pointeur de structure lui-même
	// car la structure Layer est stockée dans un tableau contigu dans NeuralNet.
	// On libère uniquement le tableau 'data' interne de la matrice/vecteur.

	deallocMatrix(&layer->weights, false);
	deallocVector(&layer->biases); // Macro ou cast vers deallocMatrix

	deallocVector(&layer->inputCache);
	deallocVector(&layer->outputCache);
	deallocVector(&layer->activationCache);
}