#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// Struktur zur Speicherung der Dateistatistiken
typedef struct {
    int zeilen;
    int woerter;
    int leerzeilen;
} DateiStatistik;

// Funktion zur Initialisierung der Statistik-Struktur
void initialisiere_statistik(DateiStatistik *stats) {
    stats->zeilen = 0;
    stats->woerter = 0;
    stats->leerzeilen = 0;
}

// Funktion zur Überprüfung, ob eine Zeile leer ist (nur Whitespace)
int ist_leerzeile(const char *zeile) {
    for (int i = 0; zeile[i] != '\0'; i++) {
        if (!isspace(zeile[i])) {
            return 0;  // Nicht leer
        }
    }
    return 1;  // Leer
}

// Funktion zum Zählen der Wörter in einer Zeile
int zaehle_woerter_in_zeile(const char *zeile) {
    int woerter = 0;
    int im_wort = 0;
    
    // ___ABC_____DEFGHIJKL
    for (int i = 0; zeile[i] != '\0'; i++) {
        if (isspace(zeile[i])) {
            im_wort = 0;
        } else {
            if (!im_wort) {
                woerter++;
                im_wort = 1;
            }
        }
    }
    
    return woerter;
}

// Funktion zur Analyse der Textdatei
int analysiere_datei(const char *dateiname, DateiStatistik *stats) {

    FILE *datei = fopen(dateiname, "r");
    
    if (datei == NULL) {
        printf("Fehler: Datei '%s' konnte nicht geöffnet werden!\n", dateiname);
        return 0;
    }
    
    char zeile[1024];
    
    // Datei zeilenweise einlesen
    while (fgets(zeile, sizeof(zeile), datei) != NULL) {
        stats->zeilen++;
        
        // Entferne Zeilenumbruch am Ende, falls vorhanden
        size_t len = strlen(zeile);
        if (len > 0 && zeile[len-1] == '\n') {
            zeile[len-1] = '\0';
        }
        
        // Prüfe ob Leerzeile
        if (ist_leerzeile(zeile)) {
            stats->leerzeilen++;
        }
        
        // Zähle Wörter in dieser Zeile
        stats->woerter += zaehle_woerter_in_zeile(zeile);
    }
    
    fclose(datei);
    return 1;
}

// Funktion zur Ausgabe der Statistik
void zeige_statistik(const DateiStatistik *stats, const char *dateiname) {
    printf("\n========================================\n");
    printf("Statistik für Datei: %s\n", dateiname);
    printf("========================================\n");
    printf("Anzahl Zeilen:      %d\n", stats->zeilen);
    printf("Anzahl Wörter:      %d\n", stats->woerter);
    printf("Anzahl Leerzeilen:  %d\n", stats->leerzeilen);
    printf("========================================\n");
}

int main(int argc, char *argv[]) {
    // Prüfe Kommandozeilenargumente
    if (argc != 2) {
        printf("Verwendung: %s <dateiname>\n", argv[0]);
        printf("Beispiel: %s test.txt\n", argv[0]);
        return 1;
    }
    
    // Erstelle und initialisiere Statistik-Struktur
    DateiStatistik stats;
    initialisiere_statistik(&stats);
    
    // Analysiere die Datei
    if (analysiere_datei(argv[1], &stats)) {
        // Zeige Ergebnisse
        zeige_statistik(&stats, argv[1]);
    } else {
        return 1;
    }
    
    return 0;
}