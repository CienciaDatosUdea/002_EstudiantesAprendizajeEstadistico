#include <iostream>

using namespace std;

int main(){
    // Debemos generar una serie con los impares
    double sum = 0.0;
    int contador = 0;
    for(int i=1;i<10000;i=i+2){
        if(contador%2==0){
            sum += 1.0/i;
        }else{
            sum -= 1.0/i;
        }
        contador+=1;
        //cout<<"Iteracion i="<<i<<" - La fraccion es 1/"<<i<<"="<<sum<<"\n";
    }
    double pi = 4.0*sum;
    cout<<"pi: "<<pi<<"\n";
    return pi;
}