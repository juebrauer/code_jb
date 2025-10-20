#include <stdio.h>

int main()
{
    printf("Anzahl Zeilen: ");
    int rows;
    scanf("%d", &rows);

    printf("Anzahl Spalten: ");
    int columns;
    scanf("%d", &columns);


    for (int y=1; y<=rows; y++)
    {
        for (int x=1; x<=columns; x++)
        {
            printf("*");
        }
        printf("\n");
    }
}