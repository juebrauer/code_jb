#include <iostream>
#include <stdlib.h>

class Matrix
{
    public:

        Matrix(int rows, int cols)
        {
            this->rows = rows;
            this->cols = cols;

            this->data = (int*) malloc(rows*cols*sizeof(int));

            for (int y=0; y<rows; y++)
            {
                for (int x=0; x<cols; x++)
                {
                    int idx = y*this->cols + x;
                    if (y==0)
                        this->data[idx] = 1;
                    else
                        this->data[idx] = 0;
                }
            }
        }

        Matrix* add(Matrix* other)
        {
            Matrix* result = new Matrix(this->rows, this->cols);

            
            return result;
        }

        void show()
        {
            for (int y=0; y<this->rows; y++)
            {
                for (int x=0; x<this->cols; x++)
                {
                    int idx = y*this->cols + x;
                    std::cout << this->data[idx] << " ";
                }
                std::cout << std::endl;
            }
        }

    private:

        int rows;
        int cols;
        int* data;

};


int main()
{
    Matrix* m1 = new Matrix(3,5);    
    Matrix* m2 = new Matrix(3,5);

    m1->show();
    m2->show();

    Matrix* m3 = m1->add(m2);
    m3->show();
}