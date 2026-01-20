#include <stdio.h>
#include <stdlib.h>
#include "network.h"

// Helper pour afficher une prédiction
void check(NeuralNetPtr net, Value i1, Value i2) {
	Vector in = createVector(2);
	in.data[0] = i1; in.data[1] = i2;
	VectorPtr out = neuralNetForward(net, &in);
	printf("[%d, %d] -> %.5f\n", (int)i1, (int)i2, out->data[0]);
	deallocVector(&in);
}

int main() {
	const char* file = "xor_model.bin";

	printf("=== 1. ENTRAINEMENT RAPIDE ===\n");
	NeuralNet net = createNeuralNet(2);
	neuralNetAddLayer(&net, 3, ACTIVATION_SIGMOID);
	neuralNetAddLayer(&net, 1, ACTIVATION_SIGMOID);

	// On fait juste quelques updates pour bouger les poids (pas besoin que ce soit parfait)
	// C'est juste pour vérifier qu'on sauvegarde bien un état non-aléatoire
	Value inputs[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} };
	Value targets[4][1] = { {0}, {1}, {1}, {0} };
	Vector vIn = createVector(2);
	Vector vTgt = createVector(1);

	for (int i = 0; i < 5000; i++) {
		int idx = i % 4;
		vIn.data[0] = inputs[idx][0]; vIn.data[1] = inputs[idx][1];
		vTgt.data[0] = targets[idx][0];
		neuralNetForward(&net, &vIn);
		neuralNetBackward(&net, &vTgt);
		neuralNetUpdate(&net, 0.5);
	}
	deallocVector(&vIn);
	deallocVector(&vTgt);

	printf("Prédictions AVANT sauvegarde :\n");
	check(&net, 0, 1);
	check(&net, 1, 1);

	// SAUVEGARDE
	printf("\n=== 2. SAUVEGARDE ===\n");
	if (!saveNeuralNet(&net, file)) return 1;

	// LIBÉRATION TOTALE
	freeNeuralNet(&net);
	printf("Réseau libéré.\n");

	// CHARGEMENT
	printf("\n=== 3. CHARGEMENT ===\n");
	NeuralNet loadedNet = loadNeuralNet(file);

	printf("Prédictions APRÈS chargement :\n");
	check(&loadedNet, 0, 1);
	check(&loadedNet, 1, 1);

	freeNeuralNet(&loadedNet);
	return 0;
}