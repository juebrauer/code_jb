const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

rl.question('Wie viele Zeilen soll das Rechteck haben? ', (zeilen) => {
  rl.question('Wie viele Spalten soll das Rechteck haben? ', (spalten) => {
    zeilen = parseInt(zeilen);
    spalten = parseInt(spalten);

    if (isNaN(zeilen) || isNaN(spalten) || zeilen < 1 || spalten < 1) {
      console.log('Bitte gib gültige positive Zahlen ein.');
      rl.close();
      return;
    }

    for (let i = 0; i < zeilen; i++) {
      console.log('#'.repeat(spalten));
    }

    rl.close();
  });
});
