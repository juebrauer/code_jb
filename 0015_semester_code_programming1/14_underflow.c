#include <stdio.h>

int main() {

    FILE *datei;
    
    // Datei zum Schreiben öffnen
    datei = fopen("underflow.txt", "w");
    
    // Prüfen, ob das Öffnen erfolgreich war
    if (datei == NULL) {
        printf("Fehler beim Öffnen der Datei!\n");
        return 1;
    }


    double x = 1.0;

    for (int i=1; i<600; i++)
    {
        x = x / 10;
        fprintf(datei, "%03d %.600f\n", i, x);
    }

    fclose(datei);
}