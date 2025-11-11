#include <stdio.h>

void save_array(int x[], int N)
{
    FILE* f;
    f  = fopen("array_sortiert.txt", "w");
    for (int i=0; i<N; i++)
    {
        fprintf(f, "%i: %d\n", i, x[i]);
    }    
    fclose(f);
}