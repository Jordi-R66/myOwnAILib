#include "layer.h"
#include <stdlib.h>
#include <math.h> 

void initLayer(LayerPtr layer, SizeT inputSize, SizeT outputSize, ActivationType activation) {
	layer->inputSize = inputSize;
	layer->outputSize = outputSize;
	layer->activation = activation;

	// 1. Allocations de base
	layer->weights = createMatrix(outputSize, inputSize);
	layer->biases = createVector(outputSize);

	// Allocations des caches (inchangé)
	layer->inputCache = createVector(inputSize);
	layer->outputCache = createVector(outputSize);
	layer->activationCache = createVector(outputSize);

	layer->delta = createVector(outputSize);
	layer->weightsGrad = createMatrix(outputSize, inputSize);
	layer->biasesGrad = createVector(outputSize);

	// 2. Initialisation Intelligente (He vs Xavier)
	double limit = 0.0;
	Value biasInit = 0.0;

	switch (activation) {
	case ACTIVATION_RELU:
		// --- HE INITIALIZATION (Kaiming He) ---
		// Idéal pour ReLU.
		// Limite = sqrt(6 / n_in) pour une distribution uniforme.
		limit = sqrt(6.0 / (double)inputSize);

		// Biais légèrement positif pour éviter le "Dying ReLU" au démarrage
		biasInit = 0.01;
		break;

	case ACTIVATION_SIGMOID:
	case ACTIVATION_SOFTMAX:
	case ACTIVATION_NONE:
	default:
		// --- XAVIER / GLOROT INITIALIZATION ---
		// Idéal pour Sigmoid, Tanh, ou Softmax.
		// Limite = sqrt(3 / n_in) pour une distribution uniforme.
		// Note: La formule complète est souvent sqrt(6 / (n_in + n_out)), 
		// mais sqrt(3 / n_in) fonctionne très bien et est plus standard.
		limit = sqrt(3.0 / (double)inputSize);

		// Biais à 0 pour laisser la sigmoïde centrée autour de 0.5 au début
		biasInit = 0.0;
		break;
	}

	// 3. Application des poids
	for (SizeT i = 0; i < layer->weights.size; i++) {
		// Random uniforme entre 0 et 1
		double r = (double)rand() / (double)RAND_MAX;
		// Projection dans [-limit, +limit]
		layer->weights.data[i] = (r * 2.0 * limit) - limit;
	}

	// 4. Application des biais
	for (SizeT i = 0; i < layer->biases.size; i++) {
		layer->biases.data[i] = biasInit;
	}
}

void freeLayer(LayerPtr layer) {
	// Le 'false' indique qu'on ne demande pas de "Wipe" (mise à 0 sécurisée) des données avant le free.
	// Note : deallocMatrix ne libère que le buffer 'data', pas le pointeur 'weights' lui-même.
	deallocMatrix(&layer->weights, false);

	deallocVector(&layer->biases);
	deallocVector(&layer->inputCache);
	deallocVector(&layer->outputCache);
	deallocVector(&layer->activationCache);
	deallocVector(&layer->delta);
	deallocMatrix(&layer->weightsGrad, false);
	deallocVector(&layer->biasesGrad);
}