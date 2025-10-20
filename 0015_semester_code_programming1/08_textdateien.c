#include <stdio.h>

int main() {
    FILE *datei;
    
    // Datei zum Schreiben öffnen
    datei = fopen("ausgabe.txt", "w");
    
    // Prüfen, ob das Öffnen erfolgreich war
    if (datei == NULL) {
        printf("Fehler beim Öffnen der Datei!\n");
        return 1;
    }

    printf("Anzahl Zeilen: ");
    int rows;
    scanf("%d", &rows);

    printf("Anzahl Spalten: ");
    int columns;
    scanf("%d", &columns);

    for (int y=1; y<=rows; y++)
    {
        for (int x=1; x<=columns; x++)
        {
            fprintf(datei, "*");
        }
        fprintf(datei, "\n");
    }
        
    // Datei schließen
    fclose(datei);
    
    return 0;
}