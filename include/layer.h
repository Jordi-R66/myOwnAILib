#pragma once

#ifndef LAYER_INCLUDED
#define LAYER_INCLUDED 1

// On inclut les maths depuis le submodule (grâce au -Iexternal/myOwnCLib du Makefile)
#include <maths/matrices/matrix.h>
#include <maths/vectors/vectors.h>

#pragma pack(1)
// Types d'activation supportés
typedef enum {
	ACTIVATION_NONE,      // Pas d'activation (Linéaire)
	ACTIVATION_SIGMOID,   // 1 / (1 + e^-x)
	ACTIVATION_RELU,      // max(0, x)
	ACTIVATION_SOFTMAX    // exp(x_i) / sum(exp(x_j))
} ActivationType;

// Structure d'une couche Dense (Fully Connected)
typedef struct Layer {
	// ... (Champs existants: weights, biases, inputCache, outputCache, activationCache) ...
	Matrix weights;
	Vector biases;

	Vector inputCache;
	Vector outputCache;
	Vector activationCache;

	// --- Nouveaux Champs pour l'Apprentissage ---
	Vector delta;       // "dZ" : Erreur de la couche (Gradient par rapport à Z)
	Matrix weightsGrad; // "dW" : Gradient des poids accumulé
	Vector biasesGrad;  // "db" : Gradient des biais accumulé

	// ... (Méta-données: inputSize, outputSize, activation) ...
	SizeT inputSize;
	SizeT outputSize;
	ActivationType activation;
} Layer, *LayerPtr;
#pragma pack()

#define LAYER_SIZE sizeof(Layer)

// Fonctions de gestion bas niveau d'une couche
// (Généralement appelées par le Network, mais utiles pour les tests unitaires)

// Initialise une couche (alloue les poids, biais et caches)
void initLayer(LayerPtr layer, SizeT inputSize, SizeT outputSize, ActivationType activation);

// Libère toute la mémoire interne de la couche
void freeLayer(LayerPtr layer);

#endif