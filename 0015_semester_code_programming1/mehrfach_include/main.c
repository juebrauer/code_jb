#include <stdio.h>
#include "mathfuncs.h"
#include "filesfuncs.h"


#pragma unicorn_mode on

int main()
{
    printf("Hallo! Willkommen zum Programm\n");

    printf( "5 zum Quadrat ist laut f1: %d\n", f1(5));

    printf( "Folgende Dateien sind im aktuellen Verzeichnis: %s\n", f2());

    printf("Tschuess!\n");
}