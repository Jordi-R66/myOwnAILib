#pragma once

#ifndef NETWORK_INCLUDED
#define NETWORK_INCLUDED 1

#include "layer.h"

#pragma pack(1)

// Structure du Réseau de Neurones
typedef struct NeuralNet {
	Layer* layers;      // Tableau dynamique de couches
	SizeT numLayers;    // Nombre de couches ajoutées
	SizeT capacity;     // Capacité actuelle
	SizeT inputSize;    // Taille de l'entrée globale
} NeuralNet, * NeuralNetPtr;

#pragma pack()

#define NEURAL_NET_SIZE sizeof(NeuralNet)

// --- API Publique ---

/**
 * @brief Create a NeuralNet object containing an empty neural net
 * @param inputSize the desired size of the neural net
 * @return NeuralNet
 */
NeuralNet createNeuralNet(SizeT inputSize);

/**
 * @brief Adds a layer to a given NeuralNet after the last one
 * @param net pointer to the neural net
 * @param outputSize Number of neurons on the new layer
 * @param activation Activation function
 */
void neuralNetAddLayer(NeuralNetPtr net, SizeT outputSize, ActivationType activation);

/**
 * @brief Propagates the input vector through each layer
 * @param net Pointer to the neural net
 * @param input Pointer to the input vector
 * @return VectorPtr Pointer to the output vector
 */
VectorPtr neuralNetForward(NeuralNetPtr net, VectorPtr input);

// --- NOUVEAUX PROTOTYPES (CORRECTION DE L'ERREUR DE COMPILATION) ---

/**
 * @brief Backpropagates the error from the target vector
 * @param net Pointer to the neural net
 * @param target Pointer to the expected output vector (label)
 */
void neuralNetBackward(NeuralNetPtr net, VectorPtr target);

/**
 * @brief Updates weights and biases using calculated gradients
 * @param net Pointer to the neural net
 * @param learningRate The step size for gradient descent (e.g., 0.1)
 */
void neuralNetUpdate(NeuralNetPtr net, Value learningRate);


/**
 * @brief Frees the NeuralNet and all its layers
 * @param net
 */
void freeNeuralNet(NeuralNetPtr net);

#endif