#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "network.h"

#define NB_CLASSES 7

// Helper pour générer un double aléatoire entre min et max
double random_range(double min, double max) {
	double scale = rand() / (double)RAND_MAX; /* [0, 1.0] */
	return min + scale * (max - min);
}

// Génère une couleur qui correspond FORCÉMENT au label demandé
void generate_color_for_label(int label, VectorPtr vIn) {
	double r, g, b;

	switch (label) {
	case 0: // Rouge (R fort, G/B faibles)
		r = random_range(0.75, 1.0);
		g = random_range(0.0, 0.25);
		b = random_range(0.0, 0.25);
		break;
	case 1: // Vert
		r = random_range(0.0, 0.25);
		g = random_range(0.75, 1.0);
		b = random_range(0.0, 0.25);
		break;
	case 2: // Bleu
		r = random_range(0.0, 0.25);
		g = random_range(0.0, 0.25);
		b = random_range(0.75, 1.0);
		break;
	case 3: // Jaune (R+G forts)
		r = random_range(0.75, 1.0);
		g = random_range(0.75, 1.0);
		b = random_range(0.0, 0.25);
		break;
	case 4: // Cyan (G+B forts)
		r = random_range(0.0, 0.25);
		g = random_range(0.75, 1.0);
		b = random_range(0.75, 1.0);
		break;
	case 5: // Magenta (R+B forts)
		r = random_range(0.75, 1.0);
		g = random_range(0.0, 0.25);
		b = random_range(0.75, 1.0);
		break;
	case 6: // Autre (Gris, sombre, milieu)
	default:
		// On génère du gris ou du bruit
		r = random_range(0.3, 0.6);
		g = random_range(0.3, 0.6);
		b = random_range(0.3, 0.6);
		break;
	}
	vIn->data[0] = r;
	vIn->data[1] = g;
	vIn->data[2] = b;
}

int main() {
	srand(time(NULL));

	printf("=== 🎨 APPRENTISSAGE ÉQUILIBRÉ ===\n");

	// 1. Architecture simplifiée
	// 3 entrées -> 16 neurones (ReLU) -> 7 sorties (Softmax)
	// C'est LARGEMENT suffisant.
	NeuralNet net = createNeuralNet(3);
	neuralNetAddLayer(&net, 16, ACTIVATION_RELU);
	neuralNetAddLayer(&net, NB_CLASSES, ACTIVATION_SOFTMAX); // OBLIGATOIRE EN SORTIE !

	Vector vIn = createVector(3);
	Vector vTgt = createVector(NB_CLASSES);

	// 2. Training Loop ÉQUILIBRÉE
	// On fait moins d'epochs (50 000), mais chaque epoch est utile.
	int epochs = 50000;
	printf("Entraînement (%d itérations)...\n", epochs);

	for (int i = 0; i < epochs; i++) {
		// Au lieu de choisir au hasard, on alterne : 0, 1, 2, 3...
		// Comme ça, le réseau voit AUTANT de rouge que d'autre.
		int label = i % NB_CLASSES;

		// 1. On fabrique l'entrée parfaite pour ce label
		generate_color_for_label(label, &vIn);

		// 2. On prépare la cible (One-Hot)
		for (SizeT k = 0; k < vTgt.size; k++) vTgt.data[k] = 0.0;
		vTgt.data[label] = 1.0;

		// 3. Apprentissage
		neuralNetForward(&net, &vIn);
		neuralNetBackward(&net, &vTgt);
		neuralNetUpdate(&net, 0.05); // LR plus doux

		if (i % (epochs / 10) == 0) printf(".");
	}
	printf(" Terminé!\n");

	// 3. Test critique : Le Rouge Pur
	printf("\n--- TEST ROUGE PUR (1.0, 0.0, 0.0) ---\n");
	vIn.data[0] = 0.0; vIn.data[1] = 0.0; vIn.data[2] = 0.0;

	VectorPtr out = neuralNetForward(&net, &vIn);

	const char* names[] = { "Rouge", "Vert", "Bleu", "Jaune", "Cyan", "Magenta", "Autre" };
	for (int k = 0; k < NB_CLASSES; k++) {
		printf("%s: %.4f\n", names[k], out->data[k]);
	}

	// Nettoyage
	freeNeuralNet(&net);
	deallocVector(&vIn);
	deallocVector(&vTgt);

	return 0;
}