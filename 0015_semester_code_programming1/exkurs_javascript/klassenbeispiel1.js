// Basisklasse
class Raum {
  constructor(nummer, flaeche) {
    this.nummer = nummer;
    this.flaeche = flaeche;
  }

  beschreibung() {
    return `Raum ${this.nummer} mit ${this.flaeche}m²`;
  }
}

// Abgeleitete Klasse 1
class Mitarbeiterraum extends Raum {
  constructor(nummer, flaeche, mitarbeiterName, telefonnummer) {
    super(nummer, flaeche); // Ruft Konstruktor der Basisklasse auf
    this.mitarbeiterName = mitarbeiterName;
    this.telefonnummer = telefonnummer;
  }

  beschreibung() {
    return `${super.beschreibung()} - Büro von ${this.mitarbeiterName}`;
  }

  anrufen() {
    return `Rufnummer: ${this.telefonnummer}`;
  }
}

// Abgeleitete Klasse 2
class Vorlesungsraum extends Raum {
  constructor(nummer, flaeche, sitzplaetze, beamerVorhanden) {
    super(nummer, flaeche);
    this.sitzplaetze = sitzplaetze;
    this.beamerVorhanden = beamerVorhanden;
  }

  beschreibung() {
    const beamer = this.beamerVorhanden ? "mit Beamer" : "ohne Beamer";
    return `${super.beschreibung()} - ${this.sitzplaetze} Plätze, ${beamer}`;
  }

  istVerfuegbar(benoetigtePlaetze) {
    return this.sitzplaetze >= benoetigtePlaetze;
  }
}

// Verwendung:
const buero = new Mitarbeiterraum("A201", 25, "Dr. Müller", "0123-456789");
const hoersaal = new Vorlesungsraum("H101", 120, 80, true);

console.log(buero.beschreibung());
// Ausgabe: Raum A201 mit 25m² - Büro von Dr. Müller

console.log(hoersaal.beschreibung());
// Ausgabe: Raum H101 mit 120m² - 80 Plätze, mit Beamer

console.log(hoersaal.istVerfuegbar(50)); // true