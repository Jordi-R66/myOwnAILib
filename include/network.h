#pragma once

#ifndef NETWORK_INCLUDED
#define NETWORK_INCLUDED 1

#include "layer.h"

#pragma pack(1)

// Structure du Réseau de Neurones
typedef struct NeuralNet {
	Layer* layers;      // Tableau dynamique de couches
	SizeT numLayers;    // Nombre de couches ajoutées
	SizeT capacity;     // Capacité actuelle du tableau (pour l'allocation dynamique)

	SizeT inputSize;    // Taille de l'entrée globale du réseau (ex: 784 pixels)
} NeuralNet, *NeuralNetPtr;

#pragma pack()

#define NEURAL_NET_SIZE sizeof(NeuralNet)

// --- API Publique ---

// 1. Création
// Crée un réseau vide prêt à recevoir des données de taille 'inputSize'
/**
 * @brief Create a NeuralNet object containing an empty neural net
 * 
 * @param inputSize the desired sizze of the neurla net
 * @return NeuralNet 
 */
NeuralNet createNeuralNet(SizeT inputSize);

// 2. Construction
// Ajoute une couche Dense à la suite des autres
// outputSize : Nombre de neurones de cette couche
// activation : Fonction d'activation (ex: ACTIVATION_RELU)
void neuralNetAddLayer(NeuralNetPtr net, SizeT outputSize, ActivationType activation);

// 3. Prédiction (Forward Propagation)
// Fait traverser l'entrée 'input' à travers toutes les couches.
// Retourne un pointeur vers le vecteur de sortie de la dernière couche.
// (Attention : le pointeur retourné pointe vers le cache interne du réseau, ne pas le free directement)
VectorPtr neuralNetForward(NeuralNetPtr net, VectorPtr input);

// 4. Nettoyage
// Libère le réseau et toutes ses couches
void freeNeuralNet(NeuralNetPtr net);

#endif