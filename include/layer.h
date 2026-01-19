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
	// --- Paramètres (Appris par le réseau) ---
	Matrix weights;  // Matrice des Poids W : [OutputSize x InputSize]
	Vector biases;   // Vecteur des Biais B : [OutputSize x 1]

	// --- Cache pour la Backpropagation (Stockage des états) ---
	// Ces vecteurs gardent en mémoire ce qui s'est passé lors du "Forward"
	// pour pouvoir calculer les gradients lors du "Backward".
	Vector inputCache;      // "X" : L'entrée reçue par la couche
	Vector outputCache;     // "Z" : W.X + B (Résultat brut avant activation)
	Vector activationCache; // "A" : f(Z) (Résultat final après activation)

	// --- Méta-données ---
	SizeT inputSize;     // Nombre de neurones en entrée (couche précédente)
	SizeT outputSize;    // Nombre de neurones dans cette couche
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