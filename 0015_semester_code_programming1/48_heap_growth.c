#include <stdio.h>
#include <malloc.h>

int main()
{
   for (int i = 1; i <= 10; i++)
   {
      char* memory_address = malloc(5000);
      printf("New memory block starts at %p\n", memory_address);
   }
}