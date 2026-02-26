#include <CLI/CLI.hpp>
#include <linenoise.h>
#include <iostream>

int main(int argc, char** argv) {
    CLI::App app{"CLI11 Test Application"};

    std::string name;
    int age = 0;

    app.add_option("-n,--name", name, "Your name")->required();
    app.add_option("-a,--age", age, "Your age")->check(CLI::PositiveNumber);

    CLI11_PARSE(app, argc, argv);

    std::cout << "Name: " << name << "\n";
    std::cout << "Age: " << age << "\n";

    while(1){
        char* line = linenoise("ToyInfer> ");
    }

    return 0;
}