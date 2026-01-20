#include "network.h"
#include <stdlib.h>
#include <stdio.h>
#include <maths/matrices/matrix.h>
// On inclut mlMatrix.h uniquement pour les fonctions d'activation (sigmoid, etc.)
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

		// 1. Sauvegarde entrée (Mise à jour du cache)
		if (currentInput->data) {
			setVector(&layer->inputCache, currentInput->data);
		}

		// 2. Z = W * X
		// Nettoyage impératif avant appel à la lib "Safe"
		if (layer->outputCache.data) {
			deallocMatrix(&layer->outputCache, false);
		}
		matrixMultiplication(&layer->weights, &layer->inputCache, &layer->outputCache);

		// 3. Z = Z + B (Addition manuelle pour éviter alloc/free excessifs)
		for (SizeT k = 0; k < layer->outputCache.size; k++) {
			layer->outputCache.data[k] += layer->biases.data[k];
		}

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
	// Boucle inversée (Cast en int pour éviter les warnings SizeT non signé)
	for (int i = (int)net->numLayers - 1; i >= 0; i--) {
		LayerPtr layer = &net->layers[i];
		LayerPtr nextLayer = (i < (int)net->numLayers - 1) ? &net->layers[i + 1] : NULL;

		// Nettoyage préventif de delta
		if (layer->delta.data) {
			deallocMatrix(&layer->delta, false);
		}

		// --- Etape A : Calcul de l'erreur (delta) ---
		if (nextLayer == NULL) {
			// Sortie : delta = A - Target
			// On recrée le vecteur delta (la lib va allouer)
			layer->delta = createVector(layer->outputSize);
			setVector(&layer->delta, layer->activationCache.data);
			vectorAxpy(-1.0, target, &layer->delta);
		}
		else {
			// Cachée : delta = W_next^T * delta_next

			// 1. Transposition manuelle de W_next
			Matrix W_T = createMatrix(nextLayer->weights.cols, nextLayer->weights.rows);
			for (SizeT r = 0; r < nextLayer->weights.rows; r++) {
				for (SizeT c = 0; c < nextLayer->weights.cols; c++) {
					W_T.data[c * W_T.cols + r] = nextLayer->weights.data[r * nextLayer->weights.cols + c];
				}
			}

			// 2. Multiplication (Delta est vide suite au dealloc au début)
			matrixMultiplication(&W_T, &nextLayer->delta, &layer->delta);

			// 3. Nettoyage temporaire
			deallocMatrix(&W_T, false);
		}

		// Multiplication par f'(Z) (Hadamard in-place)
		Values Z = layer->outputCache.data;
		Values D = layer->delta.data;
		if (D && Z) {
			for (SizeT k = 0; k < layer->outputSize; k++) {
				Value derivative = 1.0;
				switch (layer->activation) {
				case ACTIVATION_SIGMOID: derivative = d_sigmoid(Z[k]); break;
				case ACTIVATION_RELU:    derivative = d_relu(Z[k]); break;
				default: break;
				}
				D[k] *= derivative;
			}
		}

		// --- Etape B : Calcul des Gradients (dW, db) ---

		// Nettoyage préalable de la destination des gradients
		if (layer->weightsGrad.data) {
			deallocMatrix(&layer->weightsGrad, false);
		}

		// dW = delta * A_prev^T
		// Transposition manuelle de inputCache (N x 1) -> (1 x N)
		Matrix input_T = createMatrix(layer->inputCache.cols, layer->inputCache.rows);
		for (SizeT k = 0; k < layer->inputCache.rows; k++) {
			input_T.data[k] = layer->inputCache.data[k];
		}

		// Calcul final : (Out x 1) * (1 x In) -> (Out x In)
		matrixMultiplication(&layer->delta, &input_T, &layer->weightsGrad);

		deallocMatrix(&input_T, false);

		// db = delta (Copie simple)
		setVector(&layer->biasesGrad, layer->delta.data);
	}
}

void neuralNetUpdate(NeuralNetPtr net, Value learningRate) {
	for (SizeT i = 0; i < net->numLayers; i++) {
		LayerPtr layer = &net->layers[i];

		// W = W - alpha * dW
		if (layer->weights.data && layer->weightsGrad.data) {
			SizeT sizeW = layer->weights.size;
			for (SizeT k = 0; k < sizeW; k++) {
				layer->weights.data[k] -= learningRate * layer->weightsGrad.data[k];
			}
		}

		// B = B - alpha * db
		if (layer->biases.data && layer->biasesGrad.data) {
			SizeT sizeB = layer->biases.size;
			for (SizeT k = 0; k < sizeB; k++) {
				layer->biases.data[k] -= learningRate * layer->biasesGrad.data[k];
			}
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
}

// --- Helpers pour l'écriture binaire ---
static bool _writeMatrix(FILE* f, MatrixPtr mat) {
	// On écrit les dimensions pour vérification future
	if (fwrite(&mat->rows, sizeof(SizeT), 1, f) != 1) return false;
	if (fwrite(&mat->cols, sizeof(SizeT), 1, f) != 1) return false;
	// On écrit les données brutes
	size_t elements = mat->rows * mat->cols;
	if (fwrite(mat->data, sizeof(Value), elements, f) != elements) return false;
	return true;
}

static bool _readMatrix(FILE* f, MatrixPtr mat) {
	SizeT rows, cols;
	if (fread(&rows, sizeof(SizeT), 1, f) != 1) return false;
	if (fread(&cols, sizeof(SizeT), 1, f) != 1) return false;

	// Vérification de sécurité : le fichier doit correspondre à la structure allouée
	if (rows != mat->rows || cols != mat->cols) {
		fprintf(stderr, "Error: Matrix dimension mismatch during load.\n");
		return false;
	}

	size_t elements = rows * cols;
	if (fread(mat->data, sizeof(Value), elements, f) != elements) return false;
	return true;
}

bool saveNeuralNet(NeuralNetPtr net, const char* filename) {
	FILE* f = fopen(filename, "wb"); // Write Binary
	if (!f) {
		perror("Error opening file for saving");
		return false;
	}

	// 1. Métadonnées globales
	fwrite(&net->inputSize, sizeof(SizeT), 1, f);
	fwrite(&net->numLayers, sizeof(SizeT), 1, f);

	// 2. Couches
	for (SizeT i = 0; i < net->numLayers; i++) {
		LayerPtr l = &net->layers[i];

		// Métadonnées de la couche
		fwrite(&l->outputSize, sizeof(SizeT), 1, f);
		fwrite(&l->activation, sizeof(int), 1, f); // int pour l'enum

		// Données (Poids et Biais)
		if (!_writeMatrix(f, &l->weights)) { fclose(f); return false; }
		if (!_writeMatrix(f, &l->biases)) { fclose(f); return false; }
	}

	fclose(f);
	printf("Network saved to '%s'\n", filename);
	return true;
}

NeuralNet loadNeuralNet(const char* filename) {
	NeuralNet net = { 0 }; // Init vide pour retourner safe en cas d'erreur
	FILE* f = fopen(filename, "rb"); // Read Binary
	if (!f) {
		perror("Error opening file for loading");
		return net;
	}

	SizeT inputSize, numLayers;
	if (fread(&inputSize, sizeof(SizeT), 1, f) != 1) { fclose(f); return net; }
	if (fread(&numLayers, sizeof(SizeT), 1, f) != 1) { fclose(f); return net; }

	// On recrée le squelette du réseau
	net = createNeuralNet(inputSize);

	for (SizeT i = 0; i < numLayers; i++) {
		SizeT outputSize;
		int activationInt;

		fread(&outputSize, sizeof(SizeT), 1, f);
		fread(&activationInt, sizeof(int), 1, f);

		// On ajoute la couche (cela alloue la mémoire et initialise des poids aléatoires)
		neuralNetAddLayer(&net, outputSize, (ActivationType)activationInt);

		// On écrase les poids aléatoires par ceux du fichier
		LayerPtr l = &net.layers[i];
		if (!_readMatrix(f, &l->weights) || !_readMatrix(f, &l->biases)) {
			fprintf(stderr, "Error reading layer data.\n");
			freeNeuralNet(&net);
			fclose(f);
			return (NeuralNet) { 0 };
		}
	}

	fclose(f);
	printf("Network loaded from '%s'\n", filename);
	return net;
}