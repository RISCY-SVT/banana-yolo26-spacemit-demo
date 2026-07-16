#include "banana_demo/app/application.h"
#include "banana_demo/app/options.h"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    banana_demo::AppOptions options;
    std::string error;
    const banana_demo::ParseResult parsed = banana_demo::ParseAppOptions(argc, argv, options, error);
    if (parsed == banana_demo::ParseResult::kHelp) {
        std::cout << banana_demo::BuildUsage(argv[0]);
        return 0;
    }
    if (parsed == banana_demo::ParseResult::kError) {
        std::cerr << "ERROR: " << error << "\n\n" << banana_demo::BuildUsage(argv[0]);
        return 2;
    }
    try {
        banana_demo::Application application(std::move(options));
        return application.Run();
    } catch (const std::exception& exception) {
        std::cerr << "ERROR: " << exception.what() << '\n';
        return 2;
    }
}
