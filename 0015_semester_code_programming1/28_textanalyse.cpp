#include <iostream>
#include <fstream>
#include <string>
#include <cctype>

// Klasse für die Textdatei-Analyse
class TextAnalyse {
private:
    // Mitgliedsvariablen für die Statistiken
    int zeilen;
    int woerter;
    int leerzeilen;
    std::string dateiname;
    
    // Private Methode: Prüft ob eine Zeile leer ist
    bool istLeerzeile(const std::string& zeile) const {
        for (char c : zeile) {
            if (!std::isspace(static_cast<unsigned char>(c))) {
                return false;
            }
        }
        return true;
    }
    
    // Private Methode: Zählt Wörter in einer Zeile
    int zaehleWoerterInZeile(const std::string& zeile) const {
        int woerter = 0;
        bool imWort = false;
        
        for (char c : zeile) {
            if (std::isspace(static_cast<unsigned char>(c))) {
                imWort = false;
            } else {
                if (!imWort) {
                    woerter++;
                    imWort = true;
                }
            }
        }
        
        return woerter;
    }

public:
    // Konstruktor
    TextAnalyse() : zeilen(0), woerter(0), leerzeilen(0), dateiname("") {
        std::cout << "TextAnalyse-Objekt wurde erstellt." << std::endl;
    }
    
    // Konstruktor mit Dateiname
    TextAnalyse(const std::string& datei) 
        : zeilen(0), woerter(0), leerzeilen(0), dateiname(datei) {
        std::cout << "TextAnalyse-Objekt für '" << datei << "' wurde erstellt." << std::endl;
    }
    
    // Destruktor
    ~TextAnalyse() {
        std::cout << "TextAnalyse-Objekt wird zerstört." << std::endl;
    }
    
    // Methode zum Analysieren einer Datei
    bool analysiereDatei(const std::string& datei) {
        dateiname = datei;
        return analysiereDatei();
    }
    
    // Methode zum Analysieren (verwendet gespeicherten Dateinamen)
    bool analysiereDatei() {
        if (dateiname.empty()) {
            std::cerr << "Fehler: Kein Dateiname angegeben!" << std::endl;
            return false;
        }
        
        // Statistiken zurücksetzen
        zeilen = 0;
        woerter = 0;
        leerzeilen = 0;
        
        std::ifstream datei(dateiname);
        
        if (!datei.is_open()) {
            std::cerr << "Fehler: Datei '" << dateiname 
                      << "' konnte nicht geöffnet werden!" << std::endl;
            return false;
        }
        
        std::string zeile;
        
        // Datei zeilenweise einlesen
        while (std::getline(datei, zeile)) {
            zeilen++;
            
            // Prüfe ob Leerzeile
            if (istLeerzeile(zeile)) {
                leerzeilen++;
            }
            
            // Zähle Wörter in dieser Zeile
            woerter += zaehleWoerterInZeile(zeile);
        }
        
        datei.close();
        return true;
    }
    
    // Getter-Methoden
    int getZeilen() const {
        return zeilen;
    }
    
    int getWoerter() const {
        return woerter;
    }
    
    int getLeerzeilen() const {
        return leerzeilen;
    }
    
    std::string getDateiname() const {
        return dateiname;
    }
    
    // Methode zur Ausgabe der Statistik
    void zeigeStatistik() const {
        std::cout << "\n========================================"
                  << std::endl;
        std::cout << "Statistik für Datei: " << dateiname << std::endl;
        std::cout << "========================================"
                  << std::endl;
        std::cout << "Anzahl Zeilen:      " << zeilen << std::endl;
        std::cout << "Anzahl Wörter:      " << woerter << std::endl;
        std::cout << "Anzahl Leerzeilen:  " << leerzeilen << std::endl;
        std::cout << "========================================"
                  << std::endl;
    }
    
    // Methode für detaillierte Ausgabe
    void zeigeDetailliertStatistik() const {
        zeigeStatistik();
        std::cout << "\nZusätzliche Informationen:" << std::endl;
        std::cout << "Nicht-leere Zeilen: " << (zeilen - leerzeilen) 
                  << std::endl;
        if (zeilen > 0) {
            std::cout << "Durchschnittliche Wörter pro Zeile: " 
                      << static_cast<double>(woerter) / zeilen 
                      << std::endl;
        }
        if (zeilen - leerzeilen > 0) {
            std::cout << "Durchschnittliche Wörter pro nicht-leerer Zeile: "
                      << static_cast<double>(woerter) / (zeilen - leerzeilen)
                      << std::endl;
        }
    }
};

int main(int argc, char* argv[]) {
    // Prüfe Kommandozeilenargumente
    if (argc != 2) {
        std::cout << "Verwendung: " << argv[0] << " <dateiname>" 
                  << std::endl;
        std::cout << "Beispiel: " << argv[0] << " test.txt" 
                  << std::endl;
        return 1;
    }
    
    // Erstelle TextAnalyse-Objekt mit Dateiname
    TextAnalyse analyser(argv[1]);
    
    // Analysiere die Datei
    if (analyser.analysiereDatei()) {
        // Zeige Ergebnisse
        analyser.zeigeDetailliertStatistik();
        
        // Demonstration der Getter-Methoden
        std::cout << "\nZugriff über Getter:" << std::endl;
        std::cout << "Zeilen: " << analyser.getZeilen() << std::endl;
    } else {
        return 1;
    }
    
    return 0;
}