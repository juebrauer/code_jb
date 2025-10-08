#include <stdio.h>

int main() {
    int geburtsjahr;
    int aktuelles_jahr = 2025;
    int alter;
    int tage;
    int i;

    // DRY: Don't Repeat Yourself!

    i = 1;
    printf("(%d) In welchem Jahr bist du geboren? ", i);
    scanf("%d", &geburtsjahr);
    alter = aktuelles_jahr - geburtsjahr;
    tage = alter * 365;
    printf("Du bist etwa %d Tage alt.\n\n", tage);


    i = 2;
    printf("(%d) In welchem Jahr bist du geboren? ", i);
    scanf("%d", &geburtsjahr);
    alter = aktuelles_jahr - geburtsjahr;
    tage = alter * 365;
    printf("Du bist etwa %d Tage alt.\n\n", tage);


    i = 3;
    printf("(%d) In welchem Jahr bist du geboren? ", i);
    scanf("%d", &geburtsjahr);
    alter = aktuelles_jahr - geburtsjahr;
    tage = alter * 365;
    printf("Du bist etwa %d Tage alt.\n\n", tage);


    return 0;
}
