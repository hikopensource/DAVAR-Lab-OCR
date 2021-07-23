#include <vector>
#include <random>


std::vector<float>
genActs(int size) {
    std::vector<float> arr(size);
    std::mt19937 gen(0);
    std::uniform_real_distribution<> dis(0, 1);
    for(int i = 0; i < size; ++i)
        arr[i] = dis(gen);
    return arr;
}

std::vector<int>
genLabels(int alphabet_size, int L) {
    std::vector<int> label(L);

    std::mt19937 gen(1);
    std::uniform_int_distribution<> dis(1, alphabet_size - 1);

    for(int i = 0; i < L; ++i) {
        label[i] = dis(gen);
    }
    // guarantee repeats for testing
    if (L >= 3) {
        label[L / 2] = label[L / 2 + 1];
        label[L / 2 - 1] = label[L / 2];
    }
    return label;
}

