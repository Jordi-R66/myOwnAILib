#include "network.h"
#include <stdlib.h>
#include <stdio.h>
#include <maths/matrices/mlMatrix.h> 

// --- Implémentation ---

NeuralNet createNeuralNet(SizeT inputSize) {
	NeuralNet net;
	net.inputSize = inputSize;
	net.numLayers = 0;
	net.capacity = 4;

	net.layers = (Layer*)calloc(net.capacity, sizeof(Layer));
	if (net.layers == NULL) {
		fprintf(stderr, "Error: Failed to allocate memory for neural network layers.\n");
		net.capacity = 0;
	}
	return net;
}

void neuralNetAddLayer(NeuralNetPtr net, SizeT outputSize, ActivationType activation) {
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

	LayerPtr layer = &net->layers[net->numLayers];
	SizeT currentInputSize = (net->numLayers == 0) ? net->inputSize : net->layers[net->numLayers - 1].outputSize;

	initLayer(layer, currentInputSize, outputSize, activation);
	net->numLayers++;
}

VectorPtr neuralNetForward(NeuralNetPtr net, VectorPtr input) {
	VectorPtr currentInput = input;

	for (SizeT i = 0; i < net->numLayers; i++) {
		LayerPtr layer = &net->layers[i];

		// 1. Sauvegarde entrée
		if (currentInput->data) {
			setVector(&layer->inputCache, currentInput->data);
		}

		// 2. Z = W * X
		matrixMultiplication(&layer->weights, &layer->inputCache, &layer->outputCache);

		// 3. Z = Z + B
		matrixAddition(&layer->outputCache, &layer->biases);

		// 4. A = f(Z)
		setVector(&layer->activationCache, layer->outputCache.data);
		switch (layer->activation) {
		case ACTIVATION_SIGMOID: matrixMap(&layer->activationCache, sigmoid); break;
		case ACTIVATION_RELU:    matrixMap(&layer->activationCache, relu); break;
		case ACTIVATION_SOFTMAX: vectorSoftmax(&layer->activationCache); break;
		default: break;
		}
		currentInput = &layer->activationCache;
	}
	return currentInput;
}

void neuralNetBackward(NeuralNetPtr net, VectorPtr target) {
	for (int i = net->numLayers - 1; i >= 0; i--) {
		LayerPtr layer = &net->layers[i];
		LayerPtr nextLayer = (i < net->numLayers - 1) ? &net->layers[i + 1] : NULL;

		// --- Etape A : Calcul de l'erreur (delta) ---
		if (nextLayer == NULL) {
			// Sortie : delta = A - Target
			setVector(&layer->delta, layer->activationCache.data);
			vectorAxpy(-1.0, target, &layer->delta);
		}
		else {
			// Cachée : delta = Transpose(W_next) * delta_next
			// On ne peut pas utiliser matrixMultiplicationNT ici car elle fait A * B^T
			// On a besoin de W^T * delta. On transpose donc W manuellement.

			Matrix W_T = createMatrix(nextLayer->weights.cols, nextLayer->weights.rows);
			// Transpose W_next vers W_T
			// Note: matrixTranspose doit être dispo dans myOwnCLib. 
			// Si elle n'est pas dispo, il faudra l'ajouter ou faire une boucle manuelle.
			// Supposons qu'elle existe (c'est standard). Sinon voir ci-dessous.

			// Si matrixTranspose n'existe pas, voici une boucle simple :
			for (SizeT r = 0; r < nextLayer->weights.rows; r++) {
				for (SizeT c = 0; c < nextLayer->weights.cols; c++) {
					W_T.data[c * W_T.cols + r] = nextLayer->weights.data[r * nextLayer->weights.cols + c];
				}
			}

			// Calcul delta = W_T * delta_next
			matrixMultiplication(&W_T, &nextLayer->delta, &layer->delta);

			// Libération de la matrice temporaire (juste les données)
			deallocMatrix(&W_T, false);
		}

		// Multiplication par f'(Z) (Hadamard product in-place)
		Values Z = layer->outputCache.data;
		Values D = layer->delta.data;
		for (SizeT k = 0; k < layer->outputSize; k++) {
			Value derivative = 1.0;
			switch (layer->activation) {
			case ACTIVATION_SIGMOID: derivative = d_sigmoid(Z[k]); break;
			case ACTIVATION_RELU:    derivative = d_relu(Z[k]); break;
			default: break;
			}
			D[k] *= derivative;
		}

		// --- Etape B : Calcul des Gradients (dW, db) ---
		// dW = delta * A_prev^T
		// Ici matrixMultiplicationNT est parfaite (A * B^T)
		matrixMultiplicationNT(&layer->delta, &layer->inputCache, &layer->weightsGrad);

		// db = delta
		setVector(&layer->biasesGrad, layer->delta.data);
	}
}

void neuralNetUpdate(NeuralNetPtr net, Value learningRate) {
	for (SizeT i = 0; i < net->numLayers; i++) {
		LayerPtr layer = &net->layers[i];

		// W = W - lr * dW
		SizeT sizeW = layer->weights.size;
		for (SizeT k = 0; k < sizeW; k++) {
			layer->weights.data[k] -= learningRate * layer->weightsGrad.data[k];
		}

		// B = B - lr * db
		SizeT sizeB = layer->biases.size;
		for (SizeT k = 0; k < sizeB; k++) {
			layer->biases.data[k] -= learningRate * layer->biasesGrad.data[k];
		}
	}
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