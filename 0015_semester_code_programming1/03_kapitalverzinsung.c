#include <stdio.h>

int main() {
    double kapital, zinssatz;
    int jahre = 20;    

    printf("Gib dein Startkapital ein (in Euro): ");
    scanf("%lf", &kapital);

    printf("Gib den Zinssatz ein (in %%): ");
    scanf("%lf", &zinssatz);

    printf("\nJahr\tKapital (Euro)\n");
    printf("------------------------\n");

    for (int i = 1; i <= jahre; i++) {
        kapital = kapital * (1 + zinssatz / 100);
        printf("%d\t%.2f\n", i, kapital);
    }

    return 0;
}
