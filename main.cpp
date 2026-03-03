#include <cstring>

#include "CLI/CLI.hpp"
#include "banner.h"
#include "engine.h"
#include "logger.h"
#include "options.h"

int main(int argc, char** argv) {
    init_logger();
    toyinfer::Options options{};

    CLI::App app{"ToyInfer"};
    options.options_from_cli(app);
    CLI11_PARSE(app, argc, argv);

    toyinfer::Engine engine(options);

    toyinfer::Utils::print_banner();
    engine.chat();

    return 0;
}