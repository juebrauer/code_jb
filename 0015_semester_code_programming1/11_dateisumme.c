#include <stdio.h>
#include <stdlib.h>

FILE *datei;
double zahl;
double summe = 0.0;


int main(int argc, char *argv[]) {

    if (argc != 2) {
        printf("Verwendung: %s <Dateiname>\n", argv[0]);
        return 1;
    }

    datei = fopen(argv[1], "r");
    if (datei == NULL) {
        perror("Fehler beim Öffnen der Datei");
        return 1;
    }

    while (fscanf(datei, "%lf", &zahl) == 1) {
        summe += zahl;
    }

    fclose(datei);

    printf("Die Summe aller Zahlen in %s ist: %.3f\n", argv[1], summe);

    return 0;
}
