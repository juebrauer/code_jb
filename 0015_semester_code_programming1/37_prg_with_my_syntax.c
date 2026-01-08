/* main.c */

#include "my_syntax.h"

START
   BLOCKSTART
      NUMBER n;

      WRITE "Hello World" END
      NEWLINE

      WRITE "Enter a number: " END
      INPUT "%d", &n END

      WRITE "You entered the number %d", n END
      NEWLINE

   BLOCKEND