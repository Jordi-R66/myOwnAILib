#include "network.h"
#include <stdlib.h>
#include <stdio.h>
// On inclut les fonctions d'activation définies dans myOwnCLib
#include <maths/matrices/mlMatrix.h> 

// --- Implémentation ---

NeuralNet createNeuralNet(SizeT inputSize) {
	NeuralNet net;
	net.inputSize = inputSize;
	net.numLayers = 0;
	net.capacity = 4; // Capacité initiale (arbitraire, sera agrandie si besoin)

	// Allocation du tableau dynamique de couches
	net.layers = (Layer*)calloc(net.capacity, sizeof(Layer));
	if (net.layers == NULL) {
		fprintf(stderr, "Error: Failed to allocate memory for neural network layers.\n");
		net.capacity = 0;
	}

	return net;
}

void neuralNetAddLayer(NeuralNetPtr net, SizeT outputSize, ActivationType activation) {
	// 1. Redimensionnement dynamique si le tableau est plein
	if (net->numLayers >= net->capacity) {
		SizeT newCapacity = net->capacity * 2;
		Layer* newLayers = (Layer*)realloc(net->layers, newCapacity * sizeof(Layer));

		if (newLayers == NULL) {
			fprintf(stderr, "Error: Failed to reallocate memory for layers.\n");
			return;
		}

		net->layers = newLayers;
		net->capacity = newCapacity;
	}

	// 2. Préparation de la nouvelle couche
	LayerPtr layer = &net->layers[net->numLayers];

	// L'entrée de cette couche est la sortie de la précédente (ou l'entrée du réseau si c'est la 1ère)
	SizeT currentInputSize = (net->numLayers == 0) ? net->inputSize : net->layers[net->numLayers - 1].outputSize;

	// 3. Initialisation via layer.c
	initLayer(layer, currentInputSize, outputSize, activation);

	net->numLayers++;
}

VectorPtr neuralNetForward(NeuralNetPtr net, VectorPtr input) {
	VectorPtr currentInput = input;

	for (SizeT i = 0; i < net->numLayers; i++) {
		LayerPtr layer = &net->layers[i];

		// 1. Sauvegarde de l'entrée (X) dans le cache pour la Backpropagation future
		// On copie les données car le pointeur currentInput change à chaque tour
		if (currentInput->data) {
			// Utilise setVector (de vectors.h) pour copier les valeurs
			setVector(&layer->inputCache, currentInput->data);
		}

		// 2. Calcul Z = W * X
		// outputCache (Z) recevra le résultat
		// matrixMultiplication gère la dimension (Out x In) * (In x 1) -> (Out x 1)
		matrixMultiplication(&layer->weights, &layer->inputCache, &layer->outputCache);

		// 3. Ajout du Biais : Z = Z + B
		// Les vecteurs sont des matrices, on peut utiliser matrixAddition
		matrixAddition(&layer->outputCache, &layer->biases);

		// 4. Activation : A = f(Z)
		// On copie d'abord Z dans A (activationCache) pour ne pas écraser Z
		// (Z est nécessaire pour calculer dZ lors de la backprop)
		setVector(&layer->activationCache, layer->outputCache.data);

		// Application de la fonction d'activation sur A
		switch (layer->activation) {
			case ACTIVATION_SIGMOID:
				matrixMap(&layer->activationCache, sigmoid);
				break;

			case ACTIVATION_RELU:
				matrixMap(&layer->activationCache, relu);
				break;

			case ACTIVATION_SOFTMAX:
				// Optimisé dans vectors.c
				vectorSoftmax(&layer->activationCache);
				break;

			case ACTIVATION_NONE:

			default:
				// Rien à faire, A = Z (Linéaire)
				break;
		}

		// La sortie de cette couche (A) devient l'entrée de la suivante
		currentInput = &layer->activationCache;
	}

	// Retourne le pointeur vers la sortie finale (cache de la dernière couche)
	return currentInput;
}

void freeNeuralNet(NeuralNetPtr net) {
	if (net->layers != NULL) {
		for (SizeT i = 0; i < net->numLayers; i++) {
			freeLayer(&net->layers[i]);
		}

		free(net->layers);
		net->layers = NULL;
	}

	net->numLayers = 0;
	net->capacity = 0;
	net->inputSize = 0;
}