#include<stdio.h>
int main(){
  for (int i = 0; i < 1504*1024; i++){
    printf("%d : %d\n", i, i * i);
  }
}
