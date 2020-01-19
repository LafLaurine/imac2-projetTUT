
#ifndef BOOST_LIB_VEC_HPP
#define BOOST_LIB_VEC_HPP

#pragma once



void IndexError();
bool setPythonIndex(int& i, int size);

template<class V>
struct StdItem
{
    typedef typename V::value_type T;
    static T& get(V& vec, int i) //getter NEEDS to return a non-const reference
    {
        bool valid = setPythonIndex(i, vec.size());
        if(!valid)
            IndexError();
        return vec.at(i);

    }
    static void set(V& vec, int i, T const& x)
    {
        bool valid = setPythonIndex(i, vec.size());
        if (valid)
            vec.at(i) = x;
        else
            IndexError();
    }

    static void del(V& vec, int i)
    {
        bool valid = setPythonIndex(i, vec.size());
        if (valid)
            vec.erase(vec.begin()+i);
        else
            IndexError();
    }

    static void add(V& vec, const T& x)
    {
        vec.push_back(x);
    }
};


#endif
