#include "vec.hpp"

#include <Python.h>

void IndexError()
{
    PyErr_SetString(PyExc_IndexError, "Index out of range");
}

bool setPythonIndex(int& i, int size)
{

    if(i < 0) //Python lists can be accessed from the end
        i += size;
    return i >= 0 && i < size;
}
