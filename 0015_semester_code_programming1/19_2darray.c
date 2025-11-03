#include <stdio.h>

int main()
{
    int Bild[3][5];

    int counter = 0;
    for (int y=0; y<3; y++)
        for (int x=0; x<5; x++)
        {
            Bild[y][x] = counter;
            counter++; 
        }

    for (int y=0; y<3; y++)
    {
        for (int x=0; x<5; x++)
        {
            printf("%03d ", Bild[y][x]); 
        }
        printf("\n");
    }

    for (int y=0; y<3; y++)
    {
        for (int x=0; x<5; x++)
        {
            printf("Speicheradresse für Pixel (x,y)=(%d,%d): %p\n", 
                   x,y, &(Bild[y][x]) );
        }
    }

}