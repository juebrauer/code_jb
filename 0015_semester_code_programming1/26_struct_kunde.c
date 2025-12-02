#include <stdio.h>

int main() {
    printf("Hier definieren wir einen Kunden!\n");

    typedef struct {
        char* vorname;
        char* name;
        int alter;
    } kundendatensatz;

    kundendatensatz k1;

    kundendatensatz* ptr = &k1;

    ptr->alter = 49;
    ptr->vorname = "Jürgen";
    ptr->name = "Brauer";

    printf("%s %s ist %d Jahre alt.\n",
            k1.vorname, k1.name, k1.alter);
}