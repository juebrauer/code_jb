#include <stdio.h>


int* previous_memory_address;

void count_till_ten(int counter) {
   if (counter < 10) {
      int* memory_address = &counter;
      int addrdiff = previous_memory_address-memory_address;
      printf("counter=%d. "
         "Memory address of counter is %p "
         "Diff to previous address is = %d\n",
         counter, memory_address, addrdiff);
      previous_memory_address = memory_address;

      count_till_ten(counter + 1);

      printf("\nWe are back! counter is now=%d", counter);
   }
   else {
      printf("\ncounter = %d --> end of recursion\n", counter);
   }
}

int main() {
   count_till_ten(1);

}