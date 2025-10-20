#include <stdio.h>
#include <string.h>

int main()
{
    // Variante 1
    /*
    char name[10];
    printf("Gib deinen Namen ein: ");
    scanf("%10s", name);  // liest bis zum ersten Leerzeichen
    printf("Hallo %s!\n", name);
    return 0;
    */

    // Variante 2
    /*
    char name[50];
    printf("Gib deinen Namen ein: ");
    if (fgets(name, sizeof(name), stdin) != NULL) {
        name[strlen(name)] = '\0';  // entfernt das '\n' am Ende durch Nullterminierungszeichen
        printf("Hallo %s!\n", name);
    }
    */

    /*
    char name[50];
    name[0] = 'J';
    name[1] = 'u';
    name[2] = 'e';
    name[3] = 'r';
    name[4] = 'g';
    name[5] = 'e';
    name[6] = 'n';
    name[7] = '\0';
    name[8] = '?';
    name[9] = '*';
    name[10] = '&';
    printf("Du heisst: %s\n", name);
    */

    /*
    char buff[10];
    printf("Dein Name: ");
    gets(buff);
    printf("Hallo %s!", buff);
    */


    return 0;
}
