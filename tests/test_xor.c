#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "network.h"

int main() {
	// Initialisation aléatoire
	srand(time(NULL));

	printf("\n=== 🧠 TRAINING XOR (Le Hello World des IA) ===\n");
	printf("Objectif : Apprendre la table de vérité XOR :\n");
	printf("0,0 -> 0\n0,1 -> 1\n1,0 -> 1\n1,1 -> 0\n\n");

	// 1. Création du réseau : 2 entrées -> 3 cachés -> 1 sortie
	NeuralNet net = createNeuralNet(2);
	neuralNetAddLayer(&net, 3, ACTIVATION_SIGMOID); // Couche cachée
	neuralNetAddLayer(&net, 1, ACTIVATION_SIGMOID); // Couche de sortie

	// 2. Données d'entraînement
	Value inputs[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} };
	Value targets[4][1] = { {0}, {1}, {1}, {0} };

	// Vecteurs temporaires
	Vector vIn = createVector(2);
	Vector vTgt = createVector(1);

	// 3. Boucle d'apprentissage (10 000 époques)
	printf("Entrainement en cours...\n");
	for (int i = 0; i < 10000; i++) {
		// On choisit un exemple au hasard
		int idx = rand() % 4;

		// Copie des données dans les vecteurs
		vIn.data[0] = inputs[idx][0];
		vIn.data[1] = inputs[idx][1];
		vTgt.data[0] = targets[idx][0];

		// Cycle : Forward -> Backward -> Update
		neuralNetForward(&net, &vIn);
		neuralNetBackward(&net, &vTgt);
		neuralNetUpdate(&net, 0.1); // Learning Rate = 0.1
	}

	// 4. Vérification
	printf("\n--- RÉSULTATS APRÈS ENTRAÎNEMENT ---\n");
	for (int i = 0; i < 4; i++) {
		vIn.data[0] = inputs[i][0];
		vIn.data[1] = inputs[i][1];

		VectorPtr out = neuralNetForward(&net, &vIn);

		printf("Input [%d, %d] -> Prediction: %.4f (Attendu: %d)\n",
			(int)inputs[i][0], (int)inputs[i][1],
			out->data[0], (int)targets[i][0]);
	}

	// 5. Nettoyage
	freeNeuralNet(&net);
	deallocVector(&vIn);
	deallocVector(&vTgt);

	return 0;
}