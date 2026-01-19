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

void neuralNetBackward(NeuralNetPtr net, VectorPtr target) {
	// 1. On parcourt le réseau de la fin vers le début
	for (int i = net->numLayers - 1; i >= 0; i--) {
		LayerPtr layer = &net->layers[i];
		LayerPtr nextLayer = (i < net->numLayers - 1) ? &net->layers[i + 1] : NULL;

		// --- Etape A : Calcul de l'erreur (delta) ---

		if (nextLayer == NULL) {
			// A.1 Cas Couche de Sortie (Output Layer)
			// Hypothèse : Softmax + CrossEntropy OU Sigmoid + MSE
			// Dans les deux cas simplifiés, delta = A - Y

			// delta = A (activationCache)
			setVector(&layer->delta, layer->activationCache.data);

			// delta = delta - target (via axpy : delta += -1 * target)
			vectorAxpy(-1.0, target, &layer->delta);
		} else {
			// A.2 Cas Couche Cachée (Hidden Layer)
			// Formule : delta = (W_next^T * delta_next) * f'(Z)

			// 1. Rétropropagation de l'erreur du niveau suivant
			// Utilisation de matrixMultiplicationNT (Transposée) !
			// Temp : on stocke le résultat intermédiaire dans layer->delta (pour économiser de la RAM)
			matrixMultiplicationNT(&nextLayer->weights, &nextLayer->delta, &layer->delta); // delta = W^T * delta_next

			// 2. Multiplication par la dérivée de l'activation : f'(Z)
			// On calcule f'(Z) dans un vecteur temporaire ou on le fait "in-place" si on est malin
			// Pour simplifier, appliquons la dérivée sur Z (outputCache)
			// Attention : il faut un vecteur temporaire pour f'(Z) pour ne pas écraser Z
			// Ici, on va faire une astuce : on applique f' sur outputCache, on multiplie, puis on restaure Z ?
			// Non, c'est risqué. Mieux vaut un vecteur temporaire local pour f'(Z).

			// Allocation temporaire (sur la pile si petit, sinon malloc)
			// Pour être propre avec myOwnCLib, on aurait besoin d'un buffer, mais utilisons outputCache copié
			// Supposons qu'on a un vecteur temporaire "dPrime" alloué une fois dans le réseau (à ajouter ?)
			// Pour l'instant, faisons le calcul élément par élément manuellement ou ajoutons un champ "buffer".

			// Approche simple : calculons f'(Z) * delta élément par élément
			Values Z = layer->outputCache.data;
			Values D = layer->delta.data; // Contient déjà (W^T * delta_next)
			SizeT n = layer->outputSize;

			for (SizeT k = 0; k < n; k++) {
				Value derivative = 0;
				switch (layer->activation) {
					case ACTIVATION_SIGMOID: derivative = d_sigmoid(Z[k]); break;
					case ACTIVATION_RELU:    derivative = d_relu(Z[k]); break;
					default: derivative = 1.0; break;
				}

				D[k] = D[k] * derivative; // Hadamard in-place
			}
		}

		// --- Etape B : Calcul des Gradients (dW, db) ---

		// dW = delta * A_prev^T
		// A_prev est stocké dans layer->inputCache
		// delta est (Out x 1), A_prev est (In x 1). On veut (Out x In).
		// C'est une multiplication delta * A_prev^T.
		// Comme vectors = matrices (cols=1), il faut faire attention.
		// matrixMultiplicationNT avec A=delta et B=inputCache fait (Out x 1) * (1 x In) = (Out x In)
		// C'est exactement ce qu'on veut ! (Produit extérieur)
		matrixMultiplicationNT(&layer->delta, &layer->inputCache, &layer->weightsGrad);

		// db = delta
		setVector(&layer->biasesGrad, layer->delta.data);
	}
}

void neuralNetUpdate(NeuralNetPtr net, Value learningRate) {
	for (SizeT i = 0; i < net->numLayers; i++) {
		LayerPtr layer = &net->layers[i];

		// W = W - alpha * dW
		// On utilise vectorAxpy sur les données brutes (car Matrix = gros vecteur en mémoire)
		// On cast Matrix* en Vector* pour utiliser vectorAxpy (hack propre car struct compatible en RAM ?)
		// Mieux vaut utiliser une boucle ou ajouter matrixAxpy.
		// Comme Matrix contient un pointeur 'Values data', on peut travailler dessus directement.

		SizeT sizeW = layer->weights.size;
		for (SizeT k = 0; k < sizeW; k++) {
			layer->weights.data[k] -= learningRate * layer->weightsGrad.data[k];
		}

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