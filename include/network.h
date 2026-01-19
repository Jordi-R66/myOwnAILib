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
/**
 * @brief Adds a layer to a given NeuralNet after the last one
 * 
 * @param net pointer to the neural net
 * @param outputSize Number of neurons on the new layer
 * @param activation Activation function
 */
void neuralNetAddLayer(NeuralNetPtr net, SizeT outputSize, ActivationType activation);

// 3. Prédiction (Forward Propagation)
/**
 * @brief Propagates the input vector through each layer
 * 
 * @param net Pointer to the neural net
 * @param input Pointer to the input vector
 * @return VectorPtr Pointer to the output vector
 */
VectorPtr neuralNetForward(NeuralNetPtr net, VectorPtr input);

// 4. Nettoyage

/**
 * @brief Frees the NeuralNet and all its layers
 * 
 * @param net 
 */
void freeNeuralNet(NeuralNetPtr net);

#endif