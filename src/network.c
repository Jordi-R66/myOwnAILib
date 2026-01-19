#include "network.h"
#include <stdlib.h>
#include <stdio.h>
#include <maths/matrices/mlMatrix.h> // Pour relu, sigmoid

// --- Implémentation ---

NeuralNet createNeuralNet(SizeT inputSize) {
	NeuralNet net;
	net.inputSize = inputSize;
	net.numLayers = 0;
	net.capacity = 4; // Capacité initiale arbitraire (realloc si besoin)

	// Allocation du tableau dynamique de couches
	net.layers = (Layer*)malloc(sizeof(Layer) * net.capacity);

	// Note: srand(time(NULL)) devrait être appelé une seule fois dans le main de l'utilisateur

	return net;
}

void neuralNetAddLayer(NeuralNetPtr net, SizeT outputSize, ActivationType activation) {
	// Gestion de la capacité dynamique
	if (net->numLayers >= net->capacity) {
		net->capacity *= 2;
		Layer* newLayers = (Layer*)realloc(net->layers, sizeof(Layer) * net->capacity);

		if (newLayers == NULL) {
			fprintf(stderr, "Error: Failed to reallocate memory for layers.\n");
			return;
		}

		net->layers = newLayers;
	}

	// Récupération de la prochaine couche libre
	LayerPtr layer = &net->layers[net->numLayers];

	// Détermination de la taille d'entrée (InputSize)
	// Si c'est la 1ère couche : InputSize du réseau
	// Sinon : OutputSize de la couche précédente
	SizeT layerInputSize = (net->numLayers == 0) ? net->inputSize : net->layers[net->numLayers - 1].outputSize;

	// Initialisation via la fonction de layer.c
	initLayer(layer, layerInputSize, outputSize, activation);

	net->numLayers++;
}

VectorPtr neuralNetForward(NeuralNetPtr net, VectorPtr input) {
	VectorPtr currentInput = input;

	for (SizeT i = 0; i < net->numLayers; i++) {
		LayerPtr layer = &net->layers[i];

		// 1. Sauvegarde de l'entrée (X) pour la future Backpropagation
		// On copie les données car currentInput est un pointeur qui changera
		if (currentInput->data) {
			// Suppose que inputCache a la bonne taille (garanti par initLayer)
			// On copie les valeurs brutes
			setMatrix(&layer->inputCache, currentInput->data);
		}

		// 2. Calcul Z = W * X
		// Résultat stocké dans outputCache (Z)
		matrixMultiplication(&layer->weights, &layer->inputCache, &layer->outputCache);

		// 3. Ajout du Biais : Z = Z + B
		// On utilise l'addition matricielle (les vecteurs sont des matrices)
		matrixAddition(&layer->outputCache, &layer->biases);

		// 4. Activation : A = f(Z)
		// D'abord, on copie Z dans A pour ne pas écraser Z (utile pour dZ plus tard)
		setMatrix(&layer->activationCache, layer->outputCache.data);

		// Application de la fonction d'activation
		switch (layer->activation) {
			case ACTIVATION_SIGMOID:
				matrixMap(&layer->activationCache, sigmoid);
				break;

			case ACTIVATION_RELU:
				matrixMap(&layer->activationCache, relu);
				break;

			case ACTIVATION_SOFTMAX:
				// vectorSoftmax est optimisé dans myOwnCLib/vectors.c
				vectorSoftmax(&layer->activationCache);
				break;

			case ACTIVATION_NONE:

			default:
				// Linéaire : A = Z
				break;
		}

		// La sortie de cette couche devient l'entrée de la suivante
		currentInput = &layer->activationCache;
	}

	// Retourne le pointeur vers l'activation de la dernière couche
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
}