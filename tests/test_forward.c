#include <stdio.h>
#include <stdlib.h>
#include "network.h"

// Helper pour afficher un vecteur
void printVector(const char* label, VectorPtr v) {
	printf("%s [ ", label);
	if (v->data) {
		for (SizeT i = 0; i < v->rows; i++) {
			printf("%.4f ", v->data[i]);
		}
	}
	else {
		printf("NULL ");
	}
	printf("]\n");
}

int main() {
	printf("\n=== 🧠 TEST NEURAL NET FORWARD ===\n");

	// 1. Création du réseau (Input: 2 neurones)
	printf("1. Initialisation du réseau (2 -> 3 -> 1)...\n");
	NeuralNet net = createNeuralNet(2);

	// Ajout d'une couche cachée (3 neurones, Sigmoid)
	neuralNetAddLayer(&net, 3, ACTIVATION_SIGMOID);
	printf("   -> Couche cachée ajoutée (3 neurones)\n");

	// Ajout d'une couche de sortie (1 neurone, Sigmoid)
	neuralNetAddLayer(&net, 1, ACTIVATION_SIGMOID);
	printf("   -> Couche sortie ajoutée (1 neurone)\n");

	if (net.numLayers == 2) {
		printf("✅ Nombre de couches correct.\n");
	}
	else {
		printf("❌ Erreur: %zu couches au lieu de 2.\n", (size_t)net.numLayers);
	}

	// 2. Création d'un input de test (ex: [1.0, -0.5])
	printf("\n2. Préparation de l'entrée [1.0, -0.5]...\n");
	Vector input = createVector(2);
	input.data[0] = 1.0;
	input.data[1] = -0.5;

	// 3. Propagation (Forward)
	printf("\n3. Exécution du Forward...\n");
	VectorPtr output = neuralNetForward(&net, &input);

	// 4. Vérification des résultats
	printVector("Input ", &input);
	printVector("Output", output);

	// Vérifications logiques
	bool testPassed = true;
	if (output->rows != 1) {
		printf("❌ Erreur dimension sortie: %zu (Attendu 1)\n", (size_t)output->rows);
		testPassed = false;
	}

	// Avec Sigmoid, la sortie doit être entre 0 et 1
	if (output->data[0] < 0.0 || output->data[0] > 1.0) {
		printf("❌ Erreur valeur: %.4f (Hors bornes Sigmoid [0,1])\n", output->data[0]);
		testPassed = false;
	}

	if (testPassed) {
		printf("✅ Test Forward validé (Valeurs cohérentes).\n");
	}

	// 5. Nettoyage
	printf("\n4. Nettoyage mémoire...\n");
	freeNeuralNet(&net);
	deallocVector(&input);

	printf("✅ Fin du programme.\n");
	return 0;
}