#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {

    char user_input = 'j';
    
    srand(time(NULL));

    while (user_input == 'j')
    {
        for (int i = 1; i <= 10; i++)
        {
            int zahl = (rand() % 6) + 1;
            printf("%d ", zahl);
        }    
        printf("\n");

        printf("Willst du nochmal 10 Zufallszahlen erzeugen? (j/n): ");
        scanf("%c", &user_input);

        // Eingabepuffer leeren, damit '\n' von der vorherigen Eingabe nicht stört
        while (getchar() != '\n');
    };

    return 0;
}
