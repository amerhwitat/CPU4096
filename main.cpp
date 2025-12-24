
#include "RegisterN.hpp"
#include "headers.h"

using Reg4096 = RegisterN<4096>;

int main() {
    Reg4096 a = Reg4096::fromHex("0xfffffffffffffffffffffffffffff");
    Reg4096 b = Reg4096::fromHex("0xfffffffffffffffffffffffffffff");
    Reg4096 c = a + b;
    Reg4096 d = c.mul(b);
   //
        std::cout << "c = " << c.toHex() << "\n";
        std::cout << "d = " << d.toHex() << "\n";
        if (d.msb())
        std::cout << "Negative flag set\n";
        c = c + a;
        d = c.mul(c);

    std::map<std::string, std::vector<double>> dataset = {
        {"dog", {1, 0}},
        {"cat", {1, 0}},
        {"car", {0, 1}},
        {"bus", {0, 1}}
    };

    NeuralNetwork net({260, 64, 32, 2});

    // Train
    for (int epoch = 0; epoch < 5000; epoch++) {
        for (auto& pair : dataset) {
            auto input = encode_word(pair.first);
            net.backpropagate(input, pair.second);
        }
    }

    // Guess
    for (auto& word : {"dog", "cat", "car", "bus"}) {
        auto out = net.forward(encode_word(word));
        std::cout << word << " → "
                  << (out[0] > out[1] ? "animal" : "vehicle")
                  << "\n";
    }

}

