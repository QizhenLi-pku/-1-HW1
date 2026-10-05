#include <iostream>
#include <string>
using namespace std;

int main() {
    string str, substr;
    while (cin >> str >> substr) {
        int pos = 0;
        for (int i = 1; i < str.size(); i++) {
            if (str[i] > str[pos]) {
                pos = i;
            }
        }
        str.insert(pos + 1, substr);
        cout << str << endl;
    }
    return 0;
}