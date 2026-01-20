#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "network.h"

int main() {
	srand(time(NULL));

	printf("\n=== 🧠 TRAINING XOR (Parametres Boostes) ===\n");

	// 1. Création du réseau
	NeuralNet net = createNeuralNet(2);
	// Note: Pour XOR, 3 neurones cachés sont largement suffisants
	neuralNetAddLayer(&net, 50, ACTIVATION_RELU);
	neuralNetAddLayer(&net, 50, ACTIVATION_RELU);
	neuralNetAddLayer(&net, 1, ACTIVATION_SIGMOID);

	// 2. Données
	Value inputs[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} };
	Value targets[4][1] = { {0}, {1}, {1}, {0} };

	Vector vIn = createVector(2);
	Vector vTgt = createVector(1);

	// 3. Boucle d'apprentissage
	// ON AUGMENTE : 100 000 époques, LR = 0.5
	SizeT epochs = 5000000;
	double learningRate = 0.05;

	printf("Entrainement (%zu epochs, LR=%.2f)...\n", epochs, learningRate);

	for (SizeT i = 0; i < epochs; i++) {
		int idx = rand() % 4;

		vIn.data[0] = inputs[idx][0];
		vIn.data[1] = inputs[idx][1];
		vTgt.data[0] = targets[idx][0];

		neuralNetForward(&net, &vIn);
		neuralNetBackward(&net, &vTgt);
		neuralNetUpdate(&net, learningRate);

		// Petit indicateur de vie tous les 10%
		if (i % (epochs / 10) == 0) printf(".");
	}
	printf(" Terminé!\n");

	// 4. Vérification
	printf("\n--- RÉSULTATS FINAUX ---\n");
	for (int i = 0; i < 4; i++) {
		vIn.data[0] = inputs[i][0];
		vIn.data[1] = inputs[i][1];

		VectorPtr out = neuralNetForward(&net, &vIn);

		// On arrondit la prédiction pour voir si c'est bon
		int prediction = out->data[0] > 0.5 ? 1 : 0;
		printf("[%d, %d] -> %.5f (Cible: %d) => %s\n",
			(int)inputs[i][0], (int)inputs[i][1],
			out->data[0], (int)targets[i][0],
			prediction == (int)targets[i][0] ? "✅" : "❌");
	}

	freeNeuralNet(&net);
	deallocVector(&vIn);
	deallocVector(&vTgt);

	return 0;
}